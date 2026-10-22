param([Parameter(Mandatory=$true)][string]$Python, [Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference = 'Stop'
$repeatDriver = Join-Path $PSScriptRoot 'supervise_native_repeat.py'
foreach ($repeatPath in @($Python, $Plan, $repeatDriver)) {
    if (!(Test-Path -LiteralPath $repeatPath -PathType Leaf) -or $repeatPath.Contains('"')) { throw 'Repeat launch requires existing unambiguous file paths.' }
}
$repeatEnvironment = foreach ($repeatName in @('PATH','SystemRoot','WINDIR','COMSPEC','TEMP','TMP','USERPROFILE','USERNAME','USERDOMAIN','LOGNAME','USER','LNAME','APPDATA','LOCALAPPDATA','HOMEDRIVE','HOMEPATH')) {
    $repeatValue = [Environment]::GetEnvironmentVariable($repeatName, 'Process')
    if ($null -ne $repeatValue) { $repeatName+'='+$repeatValue }
}
$repeatStartup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0; CreateFlags=[uint32]16777216; EnvironmentVariables=[string[]]$repeatEnvironment}
$repeatCommand = '"'+$Python+'" "'+$repeatDriver+'" --plan "'+$Plan+'"'
$repeatCreated = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$repeatCommand; CurrentDirectory=$PSScriptRoot; ProcessStartupInformation=$repeatStartup}
if ($repeatCreated.ReturnValue -ne 0) { throw ('Windows repeat launch failed: '+$repeatCreated.ReturnValue) }
$repeatCreated | Select-Object ProcessId,ReturnValue | ConvertTo-Json -Compress
