import subprocess, json, os, sys, platform
def sh(c):
    r = subprocess.run(['bash','-lc',c],capture_output=True,text=True); return (r.stdout+r.stderr).strip()
print("== python", sys.version.split()[0], sys.executable)
try:
    import torch
    print("== torch", torch.__version__, "hip", torch.version.hip, "cuda_avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
    p = torch.cuda.get_device_properties(0); print("   device", p.name, getattr(p,'gcnArchName',''), round(p.total_memory/2**30,1), "GiB")
    import time; a = torch.randn(8192,8192,device='cuda',dtype=torch.float16); torch.cuda.synchronize(); t=time.time()
    for _ in range(20): a@a
    torch.cuda.synchronize(); print("   fp16 matmul TFLOPS ~", round(20*2*8192**3/(time.time()-t)/1e12,1))
except Exception as e: print("torch error", e)
print("== rocm", sh("cat /opt/rocm/.info/version 2>/dev/null; ls -d /opt/rocm*"))
print("== os", sh("cat /etc/os-release | head -2"))
print("== key pkgs\n"+sh("pip list 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|triton|vllm|transformers|accelerate|jupyterlab|jupyter_server|notebook|ipykernel|numpy|flash|bitsandbytes|onnx|huggingface|datasets|peft|trl|xformers|aiter|amdsmi|pytorch-triton)' "))
print("== lab extensions\n"+sh("jupyter labextension list 2>&1 | head -30"))
print("== server extensions\n"+sh("jupyter server extension list 2>&1 | grep -E 'enabled|OK' | head -30"))
print("== kernelspecs\n"+sh("jupyter kernelspec list"))
print("== tools", sh("for t in git git-lfs curl wget docker pip uv conda nvtop htop tmux vim nano gcc hipcc amd-smi rocprof rocprofv3 huggingface-cli hf ffmpeg; do printf '%s=%s ' $t $(command -v $t >/dev/null && echo yes || echo no); done"))
print("== sudo/user", sh("id; sudo -n true 2>&1 && echo sudo_ok || echo no_sudo"))
print("== env\n"+sh("env | grep -iE 'HIP|ROCM|HSA|JUPYTER|HF_|HOME|PYTORCH|CUDA|PATH' | grep -viE 'token|secret|key' | sort"))
print("== net", sh("curl -s -o /dev/null -w 'pypi=%{http_code} ' https://pypi.org/simple/ ; curl -s -o /dev/null -w 'hf=%{http_code} ' https://huggingface.co ; curl -s -o /dev/null -w 'github=%{http_code}' https://github.com"))
print("== hf download speed", sh("cd /tmp && timeout 60 curl -sL -o /dev/null -w '%{size_download} bytes %{speed_download} B/s' https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors"))
print("== limits", sh("ulimit -a | head -5; cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/cpu.max 2>/dev/null"))
print("== starter nb", sh("python -c \"import json;nb=json.load(open('/workspace/starteramd.ipynb'));[print('---',c['cell_type'],''.join(c['source'])) for c in nb['cells']]\""))
print("== jupyter procs", sh("ps -eo pid,etime,cmd | grep -iE 'jupyter|lab' | grep -v grep | cut -c1-200"))
