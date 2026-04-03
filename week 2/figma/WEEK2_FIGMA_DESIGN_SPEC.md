# Week 2 Presentation Design Spec (Figma-Ready)

## 1) Goal
Create a clean technical slide deck for Week 2 with the same visual spirit as Week 1:
- dark technical background
- high contrast typography
- strong data-first layout
- minimal decorative elements

This spec is optimized for fast assembly in Figma.

## 2) Document Setup
- Page size: 1920x1080 (16:9)
- Grid: 12 columns
- Margins: 96 px left/right, 72 px top/bottom
- Gutter: 24 px
- Baseline spacing scale: 8, 16, 24, 32, 48

## 3) Color Tokens
- BG Primary: #0A0F1A
- BG Secondary: #10182A
- Surface: #16233A
- Text Primary: #F5F7FB
- Text Secondary: #B8C1D9
- Accent Cyan: #27D3FF
- Accent Green: #52E3A4
- Accent Orange: #FFB454
- Accent Red: #FF6B6B
- Divider: #24314D

Usage:
- Titles and key numbers: Text Primary
- Body text: Text Secondary
- Positive highlights and wins: Accent Green
- Warnings and limitations: Accent Orange or Accent Red
- Links and technical labels: Accent Cyan

## 4) Typography
Recommended fonts (choose one pair):
- Headline: Space Grotesk SemiBold
- Body/Data: IBM Plex Sans Regular/Medium

Alternative pair:
- Headline: Manrope SemiBold
- Body/Data: Inter Regular/Medium

Text styles:
- H1: 64/68, SemiBold, tracking -1%
- H2: 44/48, SemiBold
- H3: 30/36, Medium
- Body L: 26/34, Regular
- Body M: 22/30, Regular
- Caption: 18/24, Medium
- Data Big: 72/72, Bold

## 5) Reusable Components
Create these as Figma components with Auto Layout:

1. Slide Header
- Left: section number and title
- Right: Week 2 | TER HPC
- Height: 88 px

2. Metric Card
- Size: 420x220
- Title, value, subtext
- Rounded corners: 20 px
- Surface fill + 1 px divider stroke

3. Comparison Table
- Header row + striped body rows
- Row height: 52 px
- Header fill: Surface
- Body fill alternating between BG Secondary and transparent

4. Callout Pill
- Variants: Success, Warning, Critical
- Success: Accent Green text on 12% green tint
- Warning: Accent Orange text on 12% orange tint
- Critical: Accent Red text on 12% red tint

5. Pipeline Block
- Small rounded box for loop diagrams (jc/pc/ic/jr/ir)

## 6) Visual Language
- Keep charts and tables front-and-center
- Use simple thin connectors (2 px) for flow diagrams
- Avoid large icons or stock illustrations
- Use subtle radial gradient in the top-right or bottom-left only
- Keep slide density high but readable

Background recipe:
- Base: BG Primary
- Overlay: radial gradient (#1A2A4A at 20% -> transparent)
- Optional texture: 2% opacity noise

## 7) Slide-by-Slide Layout Blueprint (13 slides)

Slide 1: Title
- Left block: H1 title + subtitle
- Right block: 3 metric cards
  - Peak: 781 GF/s
  - Best ratio: 182% (8T, 1024^3)
  - Best 16T ratio: 151% (2048^3)

Slide 2: Problem Statement
- Top: equation line (C = alpha AB + beta C)
- Bottom left: objective bullets
- Bottom right: simple cubic growth visual with label 2N^3

Slide 3: Why Hard
- 3 equal columns
  - CPU throughput
  - Cache hierarchy
  - Parallelism
- Bottom callout: all three must be optimized together

Slide 4: BLIS 5-loop
- Left: vertical pipeline diagram jc->pc->ic->jr->ir
- Right: short notes and MC/KC/NC cards

Slide 5: Micro-kernels
- Full-width comparison table
- Footer strip: default kernel = 6x16, peak kernel = 6x16 ASM

Slide 6: TASK1 (2D)
- Left: tile-space sketch (ic, jt)
- Right: three bullets and a big callout: pre-pack A once
- Bottom badge: up to ~2x speedup vs early Week 2

Slide 7: TASK2 (3D)
- Top: K-slice replication diagram
- Bottom: two-column Pros vs Limits

Slide 8: Auto-tuning
- Left: workflow (benchmark -> tuner -> new config)
- Right: boxes for Gradient Descent and Bayesian Opt
- Footer: adaptive-N stop criterion sigma/median < 2%

Slide 9: 5 Critical Bugs
- Two-column list with compact cards
- Each card: bug title + one-line fix
- Color critical cards in red-tinted surface

Slide 10: 8 Threads Results
- Main: 4-row table (256^3, 512^3, 1024^3, 2048^3)
- Right callout card: 495 GF/s, 182%
- Footnote: OpenBLAS wins at 2048^3 on 8T

Slide 11: 16 Threads Results
- Main: 3-row table (512^3, 1024^3, 2048^3)
- Right callout card: 781 GF/s, 151%

Slide 12: Scaling 8T -> 16T
- Main comparison table
- Highlight row 2048^3 with dual color:
  - Ours x1.67 in green
  - OpenBLAS x0.93 in orange/red

Slide 13: Conclusion
- Left: completed items checklist
- Right: next steps (TASK2 reduction, small-size path, NUMA tuning)
- Bottom closing line: architecture-level decisions drove the gains

## 8) Figma Build Order (Fast Path)
1. Create a master slide frame (1920x1080) with background and header.
2. Build components: Metric Card, Table, Callout Pill, Pipeline Block.
3. Duplicate master frame 13 times.
4. Fill content using week 2/presentation_week2.md.
5. Apply consistent spacing with 8-point system.
6. Export PDF (Presentation mode).

## 9) Export Settings
- File -> Export frames to PDF
- Include all 13 slides in order
- Use vector export for text and lines (no rasterized charts if possible)

## 10) QA Checklist Before Export
- All slides aligned to 12-column grid
- No text below 18 px
- All key numbers visually emphasized
- Colors consistent with token palette
- Tables have consistent row height and padding
- Section progression is logical and readable in under 10 minutes
