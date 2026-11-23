[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIPath
)

$ErrorActionPreference = "Stop"

$comfyUI = (Resolve-Path -LiteralPath $ComfyUIPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $comfyUI "main.py") -PathType Leaf)) {
    $portableCandidates = @(
        (Join-Path $comfyUI "ComfyUI"),
        (Join-Path $comfyUI "ComfyUI_windows_portable\ComfyUI")
    )
    $nestedComfyUI = $portableCandidates |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "main.py") -PathType Leaf } |
        Select-Object -First 1

    if (-not $nestedComfyUI) {
        throw "ComfyUIPath must point to a ComfyUI checkout containing main.py or its portable bundle root: $comfyUI"
    }

    $comfyUI = (Resolve-Path -LiteralPath $nestedComfyUI).Path
}

$source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$customNodes = Join-Path $comfyUI "custom_nodes"
$destination = Join-Path $customNodes "ComfyUI-LongCat-Avatar"

if (-not (Test-Path -LiteralPath $customNodes -PathType Container)) {
    New-Item -ItemType Directory -Path $customNodes | Out-Null
}

if (Test-Path -LiteralPath $destination) {
    $item = Get-Item -LiteralPath $destination -Force
    if ($item.LinkType -eq "Junction" -and
        [IO.Path]::GetFullPath($item.Target) -eq $source) {
        Write-Output "Already installed: $destination"
        exit 0
    }

    throw "Refusing to replace the existing path: $destination"
}

New-Item -ItemType Junction -Path $destination -Target $source | Out-Null
Write-Output "Installed this checkout into ComfyUI: $destination"
Write-Output "Restart ComfyUI, then open example_workflows\longcat_avatar_video.json."
