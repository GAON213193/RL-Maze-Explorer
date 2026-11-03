# Run with PowerShell: pwsh -NoProfile -File tests/test_setup_windows.ps1
# Parse the installer but load ONLY helper functions (no install/login side effects).
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$path = Join-Path (Split-Path $PSScriptRoot) 'scripts/setup_windows.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
foreach ($function in $ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    Invoke-Expression $function.Extent.Text
}
$shell = Join-Path $PSHOME 'pwsh'
if (-not (Test-Path $shell)) { $shell = Join-Path $PSHOME 'pwsh.exe' }
if (-not (Test-Path $shell)) { $shell = Join-Path $PSHOME 'powershell.exe' }
if (-not (Test-Command $shell @('-NoProfile','-Command','exit 0'))) { throw 'Success incorrectly rejected' }
if (Test-Command $shell @('-NoProfile','-Command','exit 7')) { throw 'Failure incorrectly accepted' }
if (Test-Command $shell @('-NoProfile','-Command','[Console]::Error.WriteLine("missing dependency"); exit 1')) { throw 'Stderr failure incorrectly accepted' }
Invoke-Checked $shell @('-NoProfile','-Command','exit 0')
$caught = $false
try { Invoke-Checked $shell @('-NoProfile','-Command','exit 7') } catch { $caught = $true }
if (-not $caught) { throw 'Install failure did not stop the flow' }
if (Test-Python (Join-Path $PSScriptRoot 'missing python.exe')) { throw 'Missing Python incorrectly accepted' }
Write-Host 'Setup syntax and helper failure-handling tests passed. Windows installation itself is not exercised.'
