#!/bin/bash
# Run this script on the Colab machine to gather environment info
# Usage: bash gather_env_info.sh > env_info.txt

echo "=========================================="
echo "Environment Information Gathering Script"
echo "=========================================="
echo ""

# Check hcr_env
echo "=== HCR_ENV ==="
echo "Location: /mnt/colab_shared/shared_Envs/hcr_env"
if [ -d "/mnt/colab_shared/shared_Envs/hcr_env" ]; then
    echo "Directory exists"

    # Check for environment.yml
    if [ -f "/mnt/colab_shared/shared_Envs/hcr_env/environment.yml" ]; then
        echo ""
        echo "--- environment.yml ---"
        cat /mnt/colab_shared/shared_Envs/hcr_env/environment.yml
    fi

    # Check for requirements.txt
    if [ -f "/mnt/colab_shared/shared_Envs/hcr_env/requirements.txt" ]; then
        echo ""
        echo "--- requirements.txt ---"
        cat /mnt/colab_shared/shared_Envs/hcr_env/requirements.txt
    fi

    # Try to get pip list from the env
    echo ""
    echo "--- Installed packages (pip list) ---"
    if [ -f "/mnt/colab_shared/shared_Envs/hcr_env/bin/pip" ]; then
        /mnt/colab_shared/shared_Envs/hcr_env/bin/pip list 2>/dev/null
    elif [ -f "/mnt/colab_shared/shared_Envs/hcr_env/bin/python" ]; then
        /mnt/colab_shared/shared_Envs/hcr_env/bin/python -m pip list 2>/dev/null
    else
        echo "Could not find pip in hcr_env"
        ls -la /mnt/colab_shared/shared_Envs/hcr_env/
    fi
else
    echo "Directory NOT found"
    echo "Listing /mnt/colab_shared/shared_Envs/:"
    ls -la /mnt/colab_shared/shared_Envs/ 2>/dev/null || echo "Parent directory not found"
fi

echo ""
echo ""
echo "=== EZFISH ENV ==="
echo "Checking common locations..."

# Try multiple possible ezfish locations
for path in "/mnt/colab_shared/shared_Envs/ezfish" \
            "/mnt/colab_shared/shared_Envs/ezfish_env" \
            "/mnt/colab_shared/ezfish" \
            "/home/*/miniconda3/envs/ezfish" \
            "/home/*/anaconda3/envs/ezfish"; do

    # Handle glob patterns
    for resolved_path in $path; do
        if [ -d "$resolved_path" ]; then
            echo "Found ezfish at: $resolved_path"

            if [ -f "$resolved_path/environment.yml" ]; then
                echo ""
                echo "--- environment.yml ---"
                cat "$resolved_path/environment.yml"
            fi

            if [ -f "$resolved_path/requirements.txt" ]; then
                echo ""
                echo "--- requirements.txt ---"
                cat "$resolved_path/requirements.txt"
            fi

            echo ""
            echo "--- Installed packages (pip list) ---"
            if [ -f "$resolved_path/bin/pip" ]; then
                $resolved_path/bin/pip list 2>/dev/null
            elif [ -f "$resolved_path/bin/python" ]; then
                $resolved_path/bin/python -m pip list 2>/dev/null
            fi
            break 2
        fi
    done
done

echo ""
echo ""
echo "=== CONDA ENVIRONMENTS LIST ==="
conda env list 2>/dev/null || echo "conda not available"

echo ""
echo ""
echo "=== AVAILABLE ENVS IN shared_Envs ==="
ls -la /mnt/colab_shared/shared_Envs/ 2>/dev/null || echo "Cannot list shared_Envs"

echo ""
echo ""
echo "=========================================="
echo "Done gathering environment info"
echo "=========================================="
