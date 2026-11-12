# Convert report/report.docx to report/report.pdf with Microsoft Word (Windows).
#   powershell -ExecutionPolicy Bypass -File report/export_pdf.ps1
# On other systems: soffice --headless --convert-to pdf report/report.docx --outdir report
$docx = Join-Path $PSScriptRoot "report.docx"
$pdf = Join-Path $PSScriptRoot "report.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0   # wdAlertsNone
try {
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    $doc.ExportAsFixedFormat($pdf, 17)   # 17 = wdExportFormatPDF
    $doc.Close($false)
    Write-Output "written $pdf"
} finally {
    $word.Quit()
}
