' Run a batch file with NO console window (Task Scheduler wrapper).
' Usage: wscript.exe run_hidden.vbs <path-to-bat>
' Waits for completion and propagates the exit code so Last Result stays honest.
Set sh = CreateObject("WScript.Shell")
rc = sh.Run("""" & WScript.Arguments(0) & """", 0, True)
WScript.Quit rc
