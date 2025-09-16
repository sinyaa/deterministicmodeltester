# PowerShell script to load environment variables from .env file
# Usage: .\load-env.ps1 [path-to-env-file]
# Default: .env in current directory

param(
    [string]$EnvFile = ".env"
)

function Load-EnvFile {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        Write-Host "ERROR: .env file not found: $FilePath" -ForegroundColor Red
        Write-Host "TIP: Create a .env file with your API keys:" -ForegroundColor Yellow
        Write-Host "   OPENAI_API_KEY=your-openai-key-here" -ForegroundColor Gray
        Write-Host "   ANTHROPIC_API_KEY=your-anthropic-key-here" -ForegroundColor Gray
        Write-Host "   GOOGLE_API_KEY=your-google-key-here" -ForegroundColor Gray
        return $false
    }
    
    Write-Host "Loading environment variables from: $FilePath" -ForegroundColor Cyan
    
    $loadedCount = 0
    $errors = @()
    
    Get-Content $FilePath | ForEach-Object {
        $line = $_.Trim()
        
        # Skip empty lines and comments
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }
        
        # Parse KEY=VALUE format
        if ($line -match "^([^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove quotes if present
            if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            elseif ($value.StartsWith("'") -and $value.EndsWith("'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            
            try {
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
                Write-Host "SUCCESS: Set $key" -ForegroundColor Green
                $loadedCount++
            }
            catch {
                $errors += "Failed to set $key : $($_.Exception.Message)"
            }
        }
        else {
            $errors += "Invalid format: $line"
        }
    }
    
    Write-Host "`nSummary:" -ForegroundColor Cyan
    Write-Host "   Variables loaded: $loadedCount" -ForegroundColor Green
    
    if ($errors.Count -gt 0) {
        Write-Host "   Errors: $($errors.Count)" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host "     - $_" -ForegroundColor Red }
    }
    
    return $loadedCount -gt 0
}

# Main execution
Write-Host "Environment Variable Loader" -ForegroundColor Magenta
Write-Host "============================" -ForegroundColor Magenta

if (Load-EnvFile $EnvFile) {
    Write-Host "`nSUCCESS: Environment variables loaded successfully!" -ForegroundColor Green
    Write-Host "TIP: You can now run your tests:" -ForegroundColor Yellow
    Write-Host "   python run_tests.py --run config/evals/languages.yaml" -ForegroundColor Gray
}
else {
    Write-Host "`nERROR: Failed to load environment variables" -ForegroundColor Red
    exit 1
}
