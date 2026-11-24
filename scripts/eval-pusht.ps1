$ErrorActionPreference = 'Stop'

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & .\.venv\Scripts\lerobot-eval.exe --policy.path=outputs/pretrained/diffusion_pusht --env.type=pusht --eval.n_episodes=1
    if ($LASTEXITCODE -ne 0) {
        throw "PushT evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
