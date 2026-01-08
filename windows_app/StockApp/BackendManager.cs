using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace StockApp;

public sealed class BackendManager
{
    public static BackendManager Instance { get; } = new BackendManager();

    private Process? _proc;
    private Process? _sovitsProc;
    private Process? _ollamaProc;
    private string? _repoRoot;

    private const string BackendStateFileName = "stockapp_backend.state.json";
    private const string OllamaStateFileName = "ollama.state.json";
    private const string SovitsStateFileName = "gpt_sovits.state.json";

    public string ApiHost { get; } = "127.0.0.1";
    public int ApiPort { get; } = 8000;

    public string ApiBase => $"http://{ApiHost}:{ApiPort}/api/v1";
    public string DashboardUrl => $"http://{ApiHost}:{ApiPort}/dashboard.html?v={Uri.EscapeDataString(DateTime.UtcNow.Ticks.ToString())}";

    // Match legacy launcher defaults
    public double SecretaryVramFrac { get; set; } = 0.30;
    public double TradingVramFrac { get; set; } = 0.70;

    // Legacy SoVITS default
    public string SovitsHost { get; set; } = "127.0.0.1";
    public int SovitsPort { get; set; } = 9880;

    private BackendManager() { }

    public async Task StartAsync()
    {
        if (_proc is { HasExited: false })
        {
            return;
        }

        _repoRoot = FindRepoRoot();
        if (string.IsNullOrWhiteSpace(_repoRoot))
        {
            throw new InvalidOperationException("Cannot locate repo root (scripts/run_live_paper_trading.py not found). Run the app from within the repo, or package the backend with the app.");
        }

        try
        {
            CleanupStaleManagedProcesses(_repoRoot);
        }
        catch
        {
        }

        var py = Path.Combine(_repoRoot, "venv311", "Scripts", "python.exe");
        if (!File.Exists(py))
        {
            py = "python";
        }

        var script = Path.Combine(_repoRoot, "scripts", "run_live_paper_trading.py");
        if (!File.Exists(script))
        {
            throw new FileNotFoundException("Backend script not found", script);
        }

        // Align performance settings (picked up by local LLM loader)
        try
        {
            Environment.SetEnvironmentVariable("SECRETARY_MAX_MEMORY_FRAC", SecretaryVramFrac.ToString(System.Globalization.CultureInfo.InvariantCulture));
            Environment.SetEnvironmentVariable("TRADING_MAX_MEMORY_FRAC", TradingVramFrac.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
        catch
        {
        }

        // Start GPT-SoVITS if configured on this machine (do not kill if already running)
        try
        {
            await EnsureSovitsAsync();
        }
        catch
        {
            // Keep app usable even if voice service fails
        }

        // If llm.mode=api, ensure Ollama is reachable before starting backend
        try
        {
            var mode = DetectLlmMode();
            if (string.Equals(mode, "api", StringComparison.OrdinalIgnoreCase))
            {
                await EnsureOllamaAsync();
            }
        }
        catch
        {
        }

        Directory.CreateDirectory(Path.Combine(_repoRoot, "logs"));
        var outLog = Path.Combine(_repoRoot, "logs", "stockapp_backend.out.log");
        var errLog = Path.Combine(_repoRoot, "logs", "stockapp_backend.err.log");
        var statePath = Path.Combine(_repoRoot, "logs", BackendStateFileName);

        var args = string.Join(' ', new[]
        {
            Quote(script),
            "--data-source", "auto",
            "--with-api",
            "--api-host", ApiHost,
            "--api-port", ApiPort.ToString(),
        });

        var psi = new ProcessStartInfo
        {
            FileName = py,
            Arguments = args,
            WorkingDirectory = _repoRoot,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        try
        {
            psi.Environment["SECRETARY_MAX_MEMORY_FRAC"] = SecretaryVramFrac.ToString(System.Globalization.CultureInfo.InvariantCulture);
            psi.Environment["TRADING_MAX_MEMORY_FRAC"] = TradingVramFrac.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        catch
        {
        }

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(outLog, ev.Data); };
        p.ErrorDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(errLog, ev.Data); };

        if (!p.Start())
        {
            throw new InvalidOperationException("Failed to start backend process");
        }

        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        _proc = p;

        try
        {
            var state = new
            {
                pid = p.Id,
                host = ApiHost,
                port = ApiPort,
                py,
                script,
                args,
                @out = outLog,
                err = errLog,
                updated_at = DateTime.UtcNow.ToString("o"),
                note = "started_by_stockapp"
            };
            File.WriteAllText(statePath, JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch
        {
        }

        var ok = await WaitApiReadyAsync(timeout: TimeSpan.FromSeconds(60));
        if (!ok)
        {
            throw new TimeoutException($"Backend did not become ready: {ApiBase}/live/status");
        }

        try
        {
            if (!DetectLlmLazyLoad())
            {
                _ = Task.Run(WarmupLocalLlmAsync);
            }
        }
        catch
        {
        }
    }

    public void Stop()
    {
        try
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(_repoRoot))
                {
                    CleanupStaleManagedProcesses(_repoRoot);
                }
            }
            catch
            {
            }

            try
            {
                if (_sovitsProc is { HasExited: false })
                {
                    try { _sovitsProc.Kill(entireProcessTree: true); } catch { }
                }
            }
            catch
            {
            }

            try
            {
                if (_ollamaProc is { HasExited: false })
                {
                    try { _ollamaProc.Kill(entireProcessTree: true); } catch { }
                }
            }
            catch
            {
            }

            if (_proc is null)
            {
                return;
            }

            if (!_proc.HasExited)
            {
                try
                {
                    _proc.CloseMainWindow();
                }
                catch
                {
                }

                try
                {
                    if (!_proc.WaitForExit(1500))
                    {
                        _proc.Kill(entireProcessTree: true);
                    }
                }
                catch
                {
                    try { _proc.Kill(entireProcessTree: true); } catch { }
                }
            }
        }
        finally
        {
            try { _proc?.Dispose(); } catch { }
            _proc = null;
            try { _sovitsProc?.Dispose(); } catch { }
            _sovitsProc = null;
            try { _ollamaProc?.Dispose(); } catch { }
            _ollamaProc = null;
        }
    }

    private static bool TcpListening(string host, int port, int timeoutMs = 300)
    {
        try
        {
            using var client = new TcpClient();
            var task = client.ConnectAsync(host, port);
            return task.Wait(timeoutMs) && client.Connected;
        }
        catch
        {
            return false;
        }
    }

    private string DetectLlmMode()
    {
        try
        {
            if (_repoRoot is null) return "";
            var cfg = Path.Combine(_repoRoot, "configs", "secretary.yaml");
            if (!File.Exists(cfg)) return "";

            var raw = File.ReadAllText(cfg, Encoding.UTF8);
            var m = Regex.Match(raw, "(?ms)^\\s*llm\\s*:\\s*.*?^\\s*mode\\s*:\\s*\"?([A-Za-z0-9_-]+)\"?");
            if (m.Success)
            {
                return m.Groups[1].Value.Trim();
            }
        }
        catch
        {
        }

        return "";
    }

    private bool DetectLlmLazyLoad()
    {
        try
        {
            if (_repoRoot is null) return false;
            var cfg = Path.Combine(_repoRoot, "configs", "secretary.yaml");
            if (!File.Exists(cfg)) return false;

            var raw = File.ReadAllText(cfg, Encoding.UTF8);
            var m = Regex.Match(raw, "(?ms)^\\s*llm\\s*:\\s*.*?^\\s*lazy_load\\s*:\\s*([A-Za-z0-9_\"-]+)");
            if (m.Success)
            {
                var v = m.Groups[1].Value.Trim().Trim('"').Trim();
                return string.Equals(v, "true", StringComparison.OrdinalIgnoreCase) || v == "1";
            }
        }
        catch
        {
        }

        return false;
    }

    private async Task EnsureOllamaAsync()
    {
        if (TcpListening("127.0.0.1", 11434, timeoutMs: 400))
        {
            return;
        }

        // Try starting ollama serve (best-effort). If user already runs it, we won't manage its lifecycle.
        string? ollamaExe = "ollama";
        try
        {
            var local = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs",
                "Ollama",
                "ollama.exe"
            );
            if (File.Exists(local))
            {
                ollamaExe = local;
            }
        }
        catch
        {
        }

        if (_repoRoot is null)
        {
            return;
        }

        Directory.CreateDirectory(Path.Combine(_repoRoot, "logs"));
        var outLog = Path.Combine(_repoRoot, "logs", "ollama.out.log");
        var errLog = Path.Combine(_repoRoot, "logs", "ollama.err.log");
        var statePath = Path.Combine(_repoRoot, "logs", OllamaStateFileName);

        var psi = new ProcessStartInfo
        {
            FileName = ollamaExe,
            Arguments = "serve",
            WorkingDirectory = _repoRoot,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(outLog, ev.Data); };
        p.ErrorDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(errLog, ev.Data); };
        try
        {
            if (!p.Start())
            {
                return;
            }
        }
        catch
        {
            return;
        }

        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        _ollamaProc = p;

        try
        {
            var state = new
            {
                pid = p.Id,
                host = "127.0.0.1",
                port = 11434,
                exe = ollamaExe,
                args = "serve",
                @out = outLog,
                err = errLog,
                updated_at = DateTime.UtcNow.ToString("o"),
                note = "started_by_stockapp"
            };
            File.WriteAllText(statePath, JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch
        {
        }

        var t0 = DateTime.UtcNow;
        while ((DateTime.UtcNow - t0).TotalSeconds < 6)
        {
            if (TcpListening("127.0.0.1", 11434, timeoutMs: 400))
            {
                break;
            }
            try
            {
                if (_ollamaProc.HasExited) break;
            }
            catch { }
            await Task.Delay(250);
        }
    }

    private async Task EnsureSovitsAsync()
    {
        if (TcpListening(SovitsHost, SovitsPort))
        {
            return;
        }

        // Legacy default locations
        var sovitsRoot = @"D:\Project\ml_cache\GPT-SoVITS";
        var sovitsPy = @"D:\Project\ml_cache\venvs\gpt_sovits\Scripts\python.exe";
        if (!Directory.Exists(sovitsRoot) || !File.Exists(sovitsPy))
        {
            return;
        }

        if (_repoRoot is null)
        {
            return;
        }

        Directory.CreateDirectory(Path.Combine(_repoRoot, "logs"));
        var outLog = Path.Combine(_repoRoot, "logs", "gpt_sovits.out.log");
        var errLog = Path.Combine(_repoRoot, "logs", "gpt_sovits.err.log");
        var statePath = Path.Combine(_repoRoot, "logs", SovitsStateFileName);

        var psi = new ProcessStartInfo
        {
            FileName = sovitsPy,
            WorkingDirectory = sovitsRoot,
            Arguments = $"api_v2.py -a {SovitsHost} -p {SovitsPort}",
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(outLog, ev.Data); };
        p.ErrorDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(errLog, ev.Data); };
        if (!p.Start())
        {
            return;
        }

        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        _sovitsProc = p;

        try
        {
            var state = new
            {
                pid = p.Id,
                host = SovitsHost,
                port = SovitsPort,
                py = sovitsPy,
                root = sovitsRoot,
                args = new[] { "api_v2.py", "-a", SovitsHost, "-p", SovitsPort.ToString() },
                @out = outLog,
                err = errLog,
                updated_at = DateTime.UtcNow.ToString("o"),
                note = "started_by_stockapp"
            };
            File.WriteAllText(statePath, JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch
        {
        }

        var t0 = DateTime.UtcNow;
        while ((DateTime.UtcNow - t0).TotalSeconds < 12)
        {
            if (TcpListening(SovitsHost, SovitsPort))
            {
                break;
            }
            try
            {
                if (_sovitsProc.HasExited) break;
            }
            catch { }
            await Task.Delay(250);
        }
    }

    private async Task WarmupLocalLlmAsync()
    {
        using var http = new HttpClient();
        var url = $"{ApiBase}/llm/warmup";
        var payload = new StringContent("{}", Encoding.UTF8, "application/json");
        var resp = await http.PostAsync(url, payload);
        // Ignore response body; warmup is best-effort
        _ = resp.IsSuccessStatusCode;
    }

    private async Task<bool> WaitApiReadyAsync(TimeSpan timeout)
    {
        var url = $"{ApiBase}/live/status";
        using var http = new HttpClient();
        var t0 = DateTime.UtcNow;

        while ((DateTime.UtcNow - t0) < timeout)
        {
            try
            {
                var resp = await http.GetAsync(url);
                if (resp.IsSuccessStatusCode)
                {
                    return true;
                }
            }
            catch
            {
            }

            try
            {
                if (_proc != null && _proc.HasExited)
                {
                    return false;
                }
            }
            catch
            {
            }

            await Task.Delay(350);
        }

        return false;
    }

    private static string Quote(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return "\"\"";
        if (s.Contains(' ')) return "\"" + s + "\"";
        return s;
    }

    private static void AppendLineSafe(string path, string line)
    {
        try
        {
            File.AppendAllText(path, line + Environment.NewLine);
        }
        catch
        {
        }
    }

    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 10 && dir is not null; i++)
        {
            var cand = Path.Combine(dir.FullName, "scripts", "run_live_paper_trading.py");
            if (File.Exists(cand))
            {
                return dir.FullName;
            }

            dir = dir.Parent;
        }

        return null;
    }

    private static void CleanupStaleManagedProcesses(string repoRoot)
    {
        try
        {
            var logsDir = Path.Combine(repoRoot, "logs");
            if (!Directory.Exists(logsDir)) return;

            TryKillFromState(Path.Combine(logsDir, BackendStateFileName));
            TryKillFromState(Path.Combine(logsDir, SovitsStateFileName));
            TryKillFromState(Path.Combine(logsDir, OllamaStateFileName));
        }
        catch
        {
        }
    }

    private static void TryKillFromState(string statePath)
    {
        try
        {
            if (!File.Exists(statePath)) return;
            var raw = File.ReadAllText(statePath, Encoding.UTF8);
            using var doc = JsonDocument.Parse(raw);
            var root = doc.RootElement;
            var note = root.TryGetProperty("note", out var noteEl) ? (noteEl.GetString() ?? "") : "";
            if (!string.Equals(note, "started_by_stockapp", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }
            if (!root.TryGetProperty("pid", out var pidEl)) return;
            var pid = pidEl.GetInt32();
            if (pid <= 0) return;

            try
            {
                using var p = Process.GetProcessById(pid);
                try { p.Kill(entireProcessTree: true); } catch { }
            }
            catch
            {
            }

            try { File.Delete(statePath); } catch { }
        }
        catch
        {
        }
    }
}
