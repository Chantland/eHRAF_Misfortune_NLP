"""
Debug Import Error - Find out WHY imports fail in specific files
Run this from the same directory as your golden_dataset_finder.py
"""

import sys
import os

print("=" * 70)
print("IMPORT ERROR DEBUGGER")
print("=" * 70)

# Check 1: Python executable
print("\n1. Python Executable:")
print(f"   {sys.executable}")
print(f"   Version: {sys.version}")

# Check 2: Virtual environment
print("\n2. Virtual Environment:")
venv = os.environ.get('VIRTUAL_ENV', 'Not activated')
print(f"   {venv}")

# Check 3: Python path
print("\n3. Python Path (sys.path):")
for i, path in enumerate(sys.path[:5], 1):
    print(f"   {i}. {path}")
if len(sys.path) > 5:
    print(f"   ... and {len(sys.path) - 5} more")

# Check 4: Working directory
print("\n4. Current Working Directory:")
print(f"   {os.getcwd()}")

# Check 5: Look for naming conflicts
print("\n5. Checking for naming conflicts...")
cwd_files = os.listdir(os.getcwd())
conflicts = []

if 'pinecone.py' in cwd_files:
    conflicts.append('pinecone.py')
    print("   ⚠️  FOUND: pinecone.py (CONFLICT!)")

if 'voyageai.py' in cwd_files:
    conflicts.append('voyageai.py')
    print("   ⚠️  FOUND: voyageai.py (CONFLICT!)")

if not conflicts:
    print("   ✅ No naming conflicts found")

# Check 6: Try importing
print("\n6. Testing imports in THIS environment:")
errors = []

try:
    import pinecone
    print(f"   ✅ import pinecone - SUCCESS")
    print(f"      Location: {pinecone.__file__}")
except ImportError as e:
    errors.append(f"pinecone: {e}")
    print(f"   ❌ import pinecone - FAILED: {e}")

try:
    from pinecone import Pinecone
    print(f"   ✅ from pinecone import Pinecone - SUCCESS")
except ImportError as e:
    errors.append(f"Pinecone: {e}")
    print(f"   ❌ from pinecone import Pinecone - FAILED: {e}")

try:
    import voyageai
    print(f"   ✅ import voyageai - SUCCESS")
    print(f"      Location: {voyageai.__file__}")
except ImportError as e:
    errors.append(f"voyageai: {e}")
    print(f"   ❌ import voyageai - FAILED: {e}")

# Check 7: Look for the problematic file
print("\n7. Checking golden_dataset_finder.py:")
if os.path.exists('golden_dataset_finder.py'):
    print("   ✅ Found golden_dataset_finder.py")

    # Try to import it
    try:
        import golden_dataset_finder
        print("   ✅ Can import golden_dataset_finder")
    except Exception as e:
        print(f"   ❌ Cannot import golden_dataset_finder: {e}")
        errors.append(f"golden_dataset_finder: {e}")
else:
    print("   ⚠️  golden_dataset_finder.py not found in current directory")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if errors:
    print("\n❌ PROBLEMS FOUND:")
    for error in errors:
        print(f"   • {error}")

    print("\n📋 SOLUTIONS:")

    if conflicts:
        print("\n1. NAMING CONFLICT DETECTED!")
        print("   You have files that conflict with package names:")
        for conflict in conflicts:
            print(f"   • {conflict}")
        print("\n   FIX: Rename these files:")
        for conflict in conflicts:
            base = conflict.replace('.py', '')
            print(f"   mv {conflict} my_{conflict}")

    if venv == 'Not activated':
        print("\n2. VIRTUAL ENVIRONMENT NOT ACTIVATED!")
        print("   FIX: Activate your virtual environment:")
        print("   source .venv/bin/activate")

    if 'golden_dataset_finder' in str(errors):
        print("\n3. CHECK YOUR IMPORTS IN golden_dataset_finder.py")
        print("   The file might have circular imports or syntax errors")
        print("   Try running: python -m py_compile golden_dataset_finder.py")
else:
    print("\n✅ EVERYTHING WORKS!")
    print("\nIf you're getting errors in PyCharm or another IDE:")
    print("1. Make sure PyCharm is using THIS Python interpreter:")
    print(f"   {sys.executable}")
    print("\n2. In PyCharm:")
    print("   - Go to Settings/Preferences")
    print("   - Python Interpreter")
    print("   - Select the .venv interpreter")
    print("   - Click OK and restart PyCharm")

print("\n" + "=" * 70)