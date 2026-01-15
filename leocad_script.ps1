$leocad = "C:\Program Files\LeoCAD\leocad.exe"
$ldrawLib = "C:\LDraw"  # or complete.zip
$inDir  = "C:\minifigs\out_ldr"
$outDir = "C:\minifigs\out_dae"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Get-ChildItem $inDir -Filter *.ldr | ForEach-Object {
  $base = $_.BaseName
  & $leocad -l $ldrawLib --export-collada "$outDir\$base.dae" $_.FullName
}