# 37 — Tiny Generative Adversarial Network

## Objective

Train a very small GAN on a synthetic 2D distribution and deploy only the trained generator on an embedded target.

## Method

A compact dense generator maps a 3-element latent vector to a 2D sample. A small discriminator is trained adversarially against it. The embedded path contains only the generator because inference does not require the discriminator.

## Files

- `train.py` — trains the GAN and exports generator weights.
- `embedded/gan_generator_demo.ino` — runs generator inference in C++.

## Run

```bash
python 37-generative-adversarial-network/train.py
```

Copy the generated `gan_generator_model.h` beside the sketch before compiling.

## TinyML relevance

This experiment separates training complexity from deployment complexity. GAN training remains desktop-side, while the embedded device runs only a small feed-forward generator.

## Limitation

The example uses synthetic 2D data to keep the model transparent. It is intended to demonstrate generator deployment, not to claim image-generation capability on a microcontroller.
