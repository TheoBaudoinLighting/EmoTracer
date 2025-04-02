# EmoTracer

*A renderer that doesn't just simulate light... it simulates feelings.*

EmoTracer is a playful path tracer written in C++17. It's not here to replace your favorite renderer. It's here to stare quietly at the floor and whisper:

> "I rendered something. I hope that's okay."

## ✨ Features

- Physically-based shading (Diffuse, Metal, Glass with absorption)
- Directional "emotional" light (blue-ish, of course)
- Cosine-weighted hemisphere sampling for smooth sad bounces
- Sadness accumulation per ray (don't worry, they can handle it)
- Shadow ray blocking, gamma correction, Fresnel-Schlick, and more
- Fully self-contained `.cpp` file (no dependencies except existential doubt)

## 🖼 Example Output

Two emotionally distant spheres, pretending it's fine:

![image](https://github.com/user-attachments/assets/c1b00616-1d49-4202-8588-1d6b65f63f59)

## 🛠 How to Build

```bash
g++ -std=c++17 -O2 emotracer.cpp -o emotracer
./emotracer
