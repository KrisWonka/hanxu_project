# Orange Pi 5 Local Dev Setup (Mac)

This repository contains a local development environment that mimics Orange Pi 5 target basics:

- Architecture: ARM64 (`linux/arm64/v8`)
- OS base: Ubuntu 22.04

## 1) Install container runtime

Option A (already set up now): Docker CLI + Colima.

```bash
brew install docker docker-compose colima
colima start --cpu 4 --memory 8 --disk 60
```

Option B: Docker Desktop for Mac.

After runtime is running, verify:

```bash
docker --version
docker-compose version
```

## 2) Build the dev image

From this project root:

```bash
docker-compose build
```

## 3) Start and enter the container

```bash
docker-compose run --rm dev
```

You should get a shell inside `/workspace` (mapped to your local project).

## 4) Useful commands

- Rebuild after Dockerfile changes:

```bash
docker-compose build --no-cache
```

- Start a one-off command:

```bash
docker-compose run --rm dev bash -lc "python3 --version && uname -m"
```

Expected architecture output is `aarch64`.

## 5) One-command workflow

Use project helper scripts:

```bash
./scripts/doctor.sh
./scripts/dev.sh --build
```

Then enter again later with:

```bash
./scripts/dev.sh
```

## 6) Notes for Orange Pi specific features

This local environment is great for:

- dependency management
- compiling code
- unit testing
- service logic development

You still need real hardware for:

- GPIO/I2C/SPI/UART interactions
- NPU/driver-specific behavior
- camera and board-level peripheral validation
