$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot
try {
    $make = Get-Command mingw32-make.exe -ErrorAction SilentlyContinue
    if (-not $make) {
        $defaultMake = "C:\msys64\ucrt64\bin\mingw32-make.exe"
        if (Test-Path $defaultMake) {
            $makePath = $defaultMake
        }
        else {
            throw "MinGW UCRT64 make was not found. Install it or add mingw32-make.exe to PATH."
        }
    }
    else {
        $makePath = $make.Source
    }

    & $makePath package-backend
    if ($LASTEXITCODE -ne 0) {
        throw "Building the static C++ backend failed with exit code $LASTEXITCODE."
    }

    & python -m PyInstaller --version
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is unavailable. Install it with: python -m pip install pyinstaller"
    }

    $specDirectory = Join-Path $projectRoot "build\pyinstaller"
    New-Item -ItemType Directory -Force -Path $specDirectory | Out-Null
    $backend = Join-Path $projectRoot "build\mp48mb-portable.exe"
    $binaryArgument = "$backend;build"

    & python -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name MP48MB `
        --specpath $specDirectory `
        --distpath (Join-Path $projectRoot "dist") `
        --workpath (Join-Path $specDirectory "work") `
        --add-binary $binaryArgument `
        (Join-Path $projectRoot "src\desktop_workbench.py")
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }

    Write-Host "Created $(Join-Path $projectRoot 'dist\MP48MB.exe')"
}
finally {
    Pop-Location
}
