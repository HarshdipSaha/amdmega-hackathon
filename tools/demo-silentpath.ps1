$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outDir = "runs/silentpath-demo-$stamp"
    python -m silentpath.cli run --matrix configs/smoke.yaml --out $outDir --cap-hours 1 --fake
    if ($LASTEXITCODE -ne 0) { throw "Fake matrix run failed with exit code $LASTEXITCODE" }

    Write-Host "`nStored comparison report:"
    python -m silentpath.cli report --out $outDir
    if ($LASTEXITCODE -ne 0) { throw "Report command failed with exit code $LASTEXITCODE" }

    Write-Host "`nMCP query functions reading the same store:"
    python -c "import json; from silentpath.mcp_server import which_path, summarise_store; p='$outDir'; print(json.dumps({'which_path': which_path(p), 'summarise_store': summarise_store(p)}, indent=2))"
    if ($LASTEXITCODE -ne 0) { throw "MCP function demo failed with exit code $LASTEXITCODE" }

    Write-Host "`nFake demo records: $outDir"
    Write-Host "Real W7900D evidence: docs/evidence/silentpath-w7900/README.md"
    Write-Host "Fake output is a software-path smoke test; it makes no GPU performance claim."
}
finally {
    Pop-Location
}
