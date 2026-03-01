# Age-to-3D Face Reconstruction Pipeline

This project implements an end-to-end pipeline that:

1. Transforms a face image to a target age using diffusion-based editing  
2. Reconstructs the aged 2D face into a 3D representation  
3. Exports a Gaussian Splatting model for interactive viewing  

The goal of this project was to move beyond simple 2D face filtering and create a system where time-altered faces can be experienced as spatial 3D objects.

---

## Overview

Pipeline:

Input Image  
→ Face Aging (Diffusion-based Editing)  
→ Aged 2D Face  
→ Single-Image 3D Reconstruction  
→ Gaussian Splat (.ply)  
→ 3D Viewer / Turntable Rendering  

---

## Stage 1: Face Aging

Model Used:  
**Face Aging via Diffusion-based Editing (FADING) / FADING_stable**

This stage performs identity-preserving age transformation using diffusion-based editing.

Key characteristics:

- Built on Stable Diffusion
- DDIM scheduler
- Null-text inversion
- Prompt-to-Prompt attention control
- Maintains identity consistency while modifying age-related features


---

## Stage 2: 2D to 3D Reconstruction

Model Used:  
**FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads**

This stage reconstructs a 3D face model from the aged 2D image.

Key characteristics:

- Single-image 3D face reconstruction
- Multi-view diffusion generation
- Gaussian Splatting-based representation
- .ply export support
- Turntable rendering capability

The final output is a Gaussian Splat representation that can be visualized interactively.

---

## Motivation

A single photograph captures only one moment in time and one viewpoint.

This project explores what happens when:
- Time is modified (aging transformation)
- Spatial dimensionality is restored (3D reconstruction)

In a dense urban environment like Chicago’s Loop, life moves quickly. 
People relocate, commute long distances, grow older, or sometimes disappear from our daily routines altogether. 

This system rethinks how we preserve and experience presence in a city defined by motion and change. It can be imagined as a tool for:

- Visualizing missing persons in age-progressed form  
- Preserving spatial memory of loved ones who have passed away  
- Reconnecting with people who are geographically distant  
- Archiving personal histories within an evolving urban landscape  

Rather than producing a filtered image, the system generates a spatial object, something that can be rotated, rendered, and experienced from multiple viewpoints.
And it's bridging time and space in a city where both are constantly in flux.

## Tech Stack

- Python 3.10
- PyTorch
- Diffusers
- Stable Diffusion
- xFormers
- FaceLift
- Gaussian Splatting
- FastAPI
- Uvicorn
- CUDA-enabled GPU (recommended)

Tested on:
- RTX 4070
- A100 (Colab)
- H100 (Colab) 

---

