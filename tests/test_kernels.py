import sys
import os

# 1. Ensure the root is in path so we can find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.cuda_loader import load_nova_kernels

def test_kernel_compilation():
    print("[*] Starting JIT Compilation of Nova Kernels...")

    # Initialize the variable in the function scope
    nova_native = None

    try:
        # This function compiles the .cu files and returns the module object
        nova_native = load_nova_kernels()

        # 2. The IF statement: Check if the module was actually loaded
        if nova_native is not None:
            print("[+] Compilation Successful!")

            # 3. Check for your specific DCNv2 function
            if hasattr(nova_native, 'dcn_forward'):
                print("[+] Bridge Successful: 'dcn_forward' is verified.")

                # TEST CALL (Optional): Verify it doesn't crash
                # result = nova_native.dcn_forward(...)
            else:
                print("[!] Warning: Module loaded, but 'dcn_forward' export is missing.")
        else:
            print("[#] Error: load_nova_kernels returned None.")

    except Exception as e:
        # This catches DLL load errors, compiler errors, etc.
        print(f"[#] Critical Failure: {str(e)}")

if __name__ == "__main__":
    test_kernel_compilation()