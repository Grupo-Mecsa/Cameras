; Instalador de DeCam para Windows (Inno Setup 6).
;
; Empaqueta la carpeta dist\DeCam que genera build_exe.py. La versión llega
; desde la línea de comandos: ISCC /DMyAppVersion=1.0.0.42 instalador.iss
;
; Se instala por usuario (sin permisos de administrador) en
; %LOCALAPPDATA%\Programs\DeCam. Encaja con dónde guarda la app su
; configuración y su log (%LOCALAPPDATA%\DeCam), y evita el problema de un
; directorio de instalación de solo lectura.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0.0"
#endif
#define MyAppName "DeCam"
#define MyAppExeName "DeCam.exe"
#define MyAppDescription "Detección de personas en la puerta a partir de grabaciones de cámaras"

[Setup]
AppId={{8E2B4C1A-7D3F-4B9E-9A1C-5D0E7F2A9B31}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
VersionInfoDescription={#MyAppDescription}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=instalador
OutputBaseFilename=DeCam-Setup-{#MyAppVersion}
; PyTorch y OpenVINO ocupan cientos de MB: lzma2 sólido reduce bastante el .exe.
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\DeCam\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
