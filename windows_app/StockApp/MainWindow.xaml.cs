using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Web.WebView2.Core;

namespace StockApp;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
        Closing += OnClosing;
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        Title = "AI Trading Terminal (Starting backend...)";

        var userData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "StockApp",
            "WebView2"
        );

        Directory.CreateDirectory(userData);

        var env = await CoreWebView2Environment.CreateAsync(null, userData);
        await WebView.EnsureCoreWebView2Async(env);

        try
        {
            await BackendManager.Instance.StartAsync();
        }
        catch (Exception ex)
        {
            Title = "AI Trading Terminal (Backend failed)";
            MessageBox.Show(ex.Message, "Backend start failed", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        var url = BackendManager.Instance.DashboardUrl;
        Title = "AI Trading Terminal";
        if (!string.IsNullOrWhiteSpace(url))
        {
            WebView.CoreWebView2.Navigate(url);
        }
    }

    private void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        try
        {
            BackendManager.Instance.Stop();
        }
        catch
        {
        }
    }
}
