using System;
using System.IO;
using System.Threading;
using System.Windows;
using System.Windows.Threading;

namespace StockApp;

public partial class App : Application
{
    private Mutex? _mutex;

    private static string GetLogPath()
    {
        try
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            for (var i = 0; i < 10 && dir is not null; i++)
            {
                var cand = Path.Combine(dir.FullName, "logs");
                var scripts = Path.Combine(dir.FullName, "scripts", "run_live_paper_trading.py");
                if (Directory.Exists(cand) && File.Exists(scripts))
                {
                    return Path.Combine(cand, "stockapp_app.log");
                }
                dir = dir.Parent;
            }
        }
        catch
        {
        }

        var local = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "StockApp",
            "stockapp_app.log"
        );
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(local) ?? Path.GetTempPath());
        }
        catch
        {
        }
        return local;
    }

    private static void Log(string msg)
    {
        try
        {
            File.AppendAllText(GetLogPath(), $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {msg}{Environment.NewLine}");
        }
        catch
        {
        }
    }

    protected override void OnStartup(StartupEventArgs e)
    {
        Log("App starting...");

        DispatcherUnhandledException += (_, ev) =>
        {
            Log("DispatcherUnhandledException: " + ev.Exception);
            try { MessageBox.Show(ev.Exception.Message, "StockApp crash", MessageBoxButton.OK, MessageBoxImage.Error); } catch { }
            ev.Handled = true;
        };

        AppDomain.CurrentDomain.UnhandledException += (_, ev) =>
        {
            Log("UnhandledException: " + (ev.ExceptionObject?.ToString() ?? "(null)"));
        };

        bool createdNew;
        _mutex = new Mutex(true, "StockApp.SingleInstance", out createdNew);
        if (!createdNew)
        {
            Log("Another instance detected -> shutting down");
            MessageBox.Show("StockApp is already running.", "StockApp", MessageBoxButton.OK, MessageBoxImage.Information);
            Shutdown();
            return;
        }

        base.OnStartup(e);
        Log("App started");
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Log("App exiting...");
        try
        {
            BackendManager.Instance.Stop();
        }
        catch
        {
        }

        try
        {
            _mutex?.ReleaseMutex();
            _mutex?.Dispose();
        }
        catch
        {
        }

        base.OnExit(e);
    }
}
