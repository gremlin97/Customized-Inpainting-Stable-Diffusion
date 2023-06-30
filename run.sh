#!/bin/bash
# python scripts/inpaint.py configs/stable-diffusion/v2-inpainting-inference.yaml 512-inpainting-ema.ckpt --b 1
#!/bin/bash

if [[ "$1" == "v2" ]]; then
    python scripts/inpaint.py configs/stable-diffusion/v2-inpainting-inference.yaml 512-inpainting-ema.ckpt --b 1
elif [[ "$1" == "v1" ]]; then
    python scripts/inpaint.py configs/stable-diffusion/v1-inpainting-inference.yaml sd-v1-5-inpainting.ckpt --b 1
else
    echo "Invalid argument. Please specify 'v1' or 'v2'."
    exit 1
fi

