# Windows PowerShell 5.1 compatible. Invoked by setup_windows.bat.
param([switch]$SkipLogin)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$logStarted = $false

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments
    if (-not $? -or $LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}
function Test-Command([string]$Exe, [string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}
function Test-Python([string]$Exe) {
    if (-not (Test-Path -LiteralPath $Exe)) { return $false }
    try {
        & $Exe -c "import sys,struct,venv,ensurepip,tkinter; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}
function Get-Python {
    $candidates = @()
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $found = & $launcher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidates += $found }
        } catch { }
    }
    $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
    $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\RL-Maze-Explorer\Python312\python.exe')
    foreach ($key in @('HKCU:\Software\Python\PythonCore\3.12\InstallPath', 'HKLM:\Software\Python\PythonCore\3.12\InstallPath')) {
        if (Test-Path $key) {
            $p = Get-ItemProperty $key
            if ($p.ExecutablePath) { $candidates += $p.ExecutablePath }
        }
    }
    # Skip Windows Store aliases, which can open the Store instead of Python.
    foreach ($cmd in @(Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
        if ($cmd.Source -and $cmd.Source -notmatch 'WindowsApps') { $candidates += $cmd.Source }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Python $candidate)) { return $candidate }
    }
    return $null
}
function Download([string]$Url, [string]$Destination) {
    Write-Host "Downloading $Url"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile "$Destination.part" -TimeoutSec 600
    Move-Item -LiteralPath "$Destination.part" -Destination $Destination -Force
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') -or ($env:PROCESSOR_ARCHITEW6432 -eq 'ARM64')) {
        throw 'This setup supports Windows 10/11 on Intel/AMD 64-bit PCs.'
    }
    foreach ($file in @('requirements-rl.txt','training\train.py','tests\test_training.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $file))) { throw 'Extract the FULL project ZIP before running setup_windows.bat.' }
    }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $cache = Join-Path $root '.setup-cache'
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    Start-Transcript -Path (Join-Path $root 'setup_windows.log') -Append | Out-Null
    $logStarted = $true
    Write-Host '[1/5] Checking Python 3.12 (64-bit, including Tk)...'
    $venv = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Python $venv)) {
        $python = Get-Python
        if (-not $python) {
            # Last traditional full Windows installer in the Python 3.12 series.
            $installer = Join-Path $cache 'python-3.12.10-amd64.exe'
            Download 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' $installer
            $signature = Get-AuthenticodeSignature -FilePath $installer
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
                throw 'Python installer signature verification failed. Installation stopped.'
            }
            $target = Join-Path $env:LOCALAPPDATA 'Programs\RL-Maze-Explorer\Python312'
            $options = @('/passive','/norestart','InstallAllUsers=0','Include_launcher=0','PrependPath=0',
                'Include_pip=1','Include_tcltk=1','Include_test=0','Include_doc=0','Shortcuts=0',
                ('TargetDir="' + $target + '"'))
            $process = Start-Process -FilePath $installer -ArgumentList $options -Wait -PassThru
            if ($process.ExitCode -notin @(0,3010)) { throw "Python installer failed: $($process.ExitCode)" }
            $python = Get-Python
            if (-not $python) { throw 'Python installation could not be verified. A restart or administrator assistance may be required.' }
        }
        if (Test-Path -LiteralPath (Join-Path $root '.venv')) {
            $backup = '.venv-backup-' + [guid]::NewGuid().ToString('N')
            Rename-Item -LiteralPath (Join-Path $root '.venv') -NewName $backup
            Write-Host "Preserved the previous environment as $backup"
        }
        Invoke-Checked $python @('-m','venv',(Join-Path $root '.venv'))
        if (-not (Test-Python $venv)) { throw 'Virtual environment validation failed.' }
    }
    Write-Host '[2/5] Installing/checking CPU training dependencies...'
    $dependenciesReady = Test-Command $venv @('-c', "from importlib.metadata import version; import torch,gymnasium,stable_baselines3,numpy; assert version('stable-baselines3')=='2.9.0'; assert version('gymnasium')=='1.4.0'; assert version('numpy')=='2.5.4'; assert version('torch').split('+')[0]=='2.14.1'")
    if (-not $dependenciesReady) {
        Invoke-Checked $venv @('-m','pip','install','torch==2.14.1','--index-url','https://download.pytorch.org/whl/cpu')
        Invoke-Checked $venv @('-m','pip','install','-r','requirements-rl.txt')
    }
    Invoke-Checked $venv @('-m','pip','check')

    Write-Host '[3/5] Preparing GitHub CLI...'
    $gh = $null
    $ghCommand = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($ghCommand) { $gh = $ghCommand.Source }
    $portable = Join-Path $root '.tools\gh\bin\gh.exe'
    if (-not $gh -and (Test-Path -LiteralPath $portable)) { $gh = $portable }
    if (-not $gh) {
        $version = '2.103.3'
        $archiveName = "gh_${version}_windows_amd64.zip"
        $archive = Join-Path $cache $archiveName
        $checksums = Join-Path $cache "gh_${version}_checksums.txt"
        $base = "https://github.com/cli/cli/releases/download/v$version"
        Download "$base/$archiveName" $archive
        Download "$base/gh_${version}_checksums.txt" $checksums
        $expected = $null
        foreach ($line in Get-Content -LiteralPath $checksums) {
            if ($line -match ('^([a-fA-F0-9]{64})\s+\*?' + [regex]::Escape($archiveName) + '$')) { $expected = $Matches[1] }
        }
        if (-not $expected -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
            throw 'GitHub CLI checksum verification failed.'
        }
        $unpack = Join-Path $cache ('gh-' + [guid]::NewGuid().ToString('N'))
        Expand-Archive -LiteralPath $archive -DestinationPath $unpack
        $binary = Get-ChildItem -LiteralPath $unpack -Filter gh.exe -Recurse | Select-Object -First 1
        if (-not $binary) { throw 'GitHub CLI archive did not contain gh.exe.' }
        New-Item -ItemType Directory -Path (Split-Path $portable) -Force | Out-Null
        Copy-Item -LiteralPath $binary.FullName -Destination $portable
        $gh = $portable
    }
    Invoke-Checked $gh @('--version')
    $env:PATH = (Split-Path $gh) + ';' + $env:PATH

    Write-Host '[4/5] Testing the environment, model training and save/load...'
    Invoke-Checked $venv @('-m','unittest','discover','-s','tests','-v')
    # No production model is trained, overwritten or uploaded by setup.
    Write-Host 'Python, training libraries and GitHub CLI are ready.'
    Stop-Transcript | Out-Null
    $logStarted = $false
    Write-Host '[5/5] GitHub login (complete browser authorization yourself)...'
    if (-not $SkipLogin) {
        if (-not (Test-Command $gh @('auth','status','--hostname','github.com'))) {
            & $gh auth login --hostname github.com --git-protocol https --web
            if ($LASTEXITCODE -ne 0) {
                Write-Host 'Software setup succeeded. GitHub login is incomplete; train_ppo.bat will retry login.'
                exit 2
            }
        }
    }
    Write-Host 'SETUP COMPLETE. Open train_ppo.bat to start or continue training.'
    exit 0
} catch {
    Write-Host ('SETUP FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Check setup_windows.log, fix the issue and run setup_windows.bat again.'
    Write-Host 'Your models and training records have not been deleted.'
    exit 1
} finally {
    if ($logStarted) { Stop-Transcript | Out-Null }
}
