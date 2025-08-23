# Auto-install datastores dependencies from pyproject.toml
import sys
import subprocess
import importlib.util
from pathlib import Path

def get_project_root():
    """Find the project root containing pyproject.toml"""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None

def install_optional_dependencies(group_name="database"):
    """Install optional dependencies for a specific group"""
    project_root = get_project_root()
    if not project_root:
        print("Could not find pyproject.toml. Manual installation required:")
        print(f"pip install engforge[{group_name}]")
        return False
    
    try:
        # Try to import tomllib (Python 3.11+) or tomli (fallback)
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        # Read pyproject.toml
        with open(project_root / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        
        optional_deps = config.get("project", {}).get("optional-dependencies", {})
        if group_name not in optional_deps:
            print(f"No optional dependency group '{group_name}' found")
            return False
        
        # Install the optional dependencies
        package_name = config.get("project", {}).get("name", "engforge")
        install_spec = f"{package_name}[{group_name}]"
        
        print(f"Installing {install_spec}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", install_spec
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {install_spec}")
            return True
        else:
            print(f"❌ Failed to install {install_spec}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error during auto-installation: {e}")
        print(f"Please install manually: pip install engforge[{group_name}]")
        return False

def check_and_install_datastores():
    """Check if datastores dependencies are available, auto-install if needed"""
    try:
        import engforge.datastores.data
        return True
    except ImportError as e:
        print(f"Datastores dependencies not available: {e}")
        print("")
        
        # Check if we're in development mode (editable install) and interactive
        project_root = get_project_root()
        is_interactive = hasattr(sys, 'ps1') or sys.stdout.isatty()
        
        if project_root and (project_root / "pyproject.toml").exists() and is_interactive:
            try:
                answer = input("Auto-install database dependencies? (y/N): ").strip().lower()
                if answer in ['y', 'yes']:
                    if install_optional_dependencies("database"):
                        print("Please restart your Python session to use the newly installed dependencies.")
                        return False
                    else:
                        print("Auto-installation failed. Install manually with:")
                        print("pip install engforge[database]")
                        return False
                else:
                    print("Install database dependencies with:")
                    print("pip install engforge[database]")
                    return False
            except (EOFError, KeyboardInterrupt):
                print("Install database dependencies with:")
                print("pip install engforge[database]")
                return False
        else:
            print("Install database dependencies with:")
            print("pip install engforge[database]")
            return False

# Auto-check and install on import
if not check_and_install_datastores():
    pass  # Dependencies not available, but user has been informed
