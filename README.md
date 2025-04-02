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

![image](https://github.com/user-attachments/assets/2f32404f-7b2f-4dc9-8393-06f123fae1c9)

## 💠 How to Build

You can compile EmoTracer with any modern C++17-compatible compiler. Example with g++:

```bash
g++ -std=c++17 -O2 emotracer.cpp -o emotracer
./emotracer
```

**If you're using Visual Studio (.sln)**: just include `emotracer.cpp` in your solution, set the entry point to `main`, compile in Release mode, and let the melancholy unfold.

Output will be a PPM file: `emotracer.ppm`. You can open it with any image viewer or convert it to PNG using ImageMagick:

```bash
convert emotracer.ppm emotracer.png
```

## 🧠 Philosophy

> EmoTracer is not about fast rendering.  
> It's about slow convergence — both in light transport and personal growth.

Perfect for graphics students, curious devs, and emotionally stable individuals who enjoy poetic code.

## 🤝 Contributions

Open an issue, submit a pull request, or just sit in the dark and reflect. It's all valid.

---

Made with ☕, 💡 by [Théo Baudoin](https://github.com/TheoBaudoinLighting)
