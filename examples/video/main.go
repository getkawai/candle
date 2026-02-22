package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/soundprediction/go-candle/pkg/candle"
)

func main() {
	if err := candle.Init(); err != nil {
		log.Fatalf("Failed to initialize candle: %v", err)
	}
	fmt.Println("candle version:", candle.Version())

	if !candle.IsVideoAvailable() {
		log.Fatal("Video pipeline is not available. Please rebuild the candle binding with video support.")
	}

	modelID := flag.String("model", "Lightricks/LTX-Video-2b-v0.9", "HuggingFace model ID for video generation")
	cacheDir := flag.String("cache", "", "Optional cache directory for model weights")
	outputDir := flag.String("output", "./output", "Output directory for generated video frames")
	outputGIF := flag.String("gif", "", "Output path for GIF file (optional)")
	prompt := flag.String("prompt", "A cat playing piano in a cozy living room, cinematic lighting", "Text prompt for video generation")
	height := flag.Int("height", 512, "Video height (must be divisible by 32)")
	width := flag.Int("width", 704, "Video width (must be divisible by 32)")
	numFrames := flag.Int("frames", 65, "Number of frames to generate")
	steps := flag.Int("steps", 30, "Number of inference steps")
	guidance := flag.Float64("guidance", 3.0, "Guidance scale")
	fps := flag.Int("fps", 24, "Frames per second for output")
	seed := flag.Uint64("seed", 0, "Random seed (0 for random)")
	flag.Parse()

	if *height%32 != 0 || *width%32 != 0 {
		log.Fatal("Height and width must be divisible by 32")
	}

	fmt.Printf("\n--- Video Generation ---\n")
	fmt.Printf("Model: %s\n", *modelID)
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Resolution: %dx%d, %d frames @ %d fps\n", *width, *height, *numFrames, *fps)
	fmt.Printf("Steps: %d, Guidance: %.1f\n", *steps, *guidance)

	cfg := candle.VideoConfig{
		ModelID:  *modelID,
		CacheDir: *cacheDir,
	}

	fmt.Println("\nLoading video pipeline...")
	pipeline, err := candle.NewVideoPipeline(cfg)
	if err != nil {
		log.Fatalf("Failed to create video pipeline: %v", err)
	}
	defer pipeline.Close()
	fmt.Println("Pipeline loaded successfully")

	params := candle.VideoGenerationParams{
		Height:            *height,
		Width:             *width,
		NumFrames:         *numFrames,
		NumInferenceSteps: *steps,
		GuidanceScale:     float32(*guidance),
		FrameRate:         *fps,
		Seed:              *seed,
	}

	fmt.Println("\nGenerating video...")
	result, err := pipeline.Generate(*prompt, params)
	if err != nil {
		log.Fatalf("Video generation failed: %v", err)
	}
	defer result.Close()

	fmt.Printf("Generated %d frames at %d fps\n", result.FrameCount, result.FPS)

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	fmt.Printf("\nSaving frames to %s...\n", *outputDir)
	if err := result.SaveFrames(*outputDir); err != nil {
		log.Printf("Warning: Failed to save frames: %v", err)
	} else {
		fmt.Printf("Frames saved to %s/\n", *outputDir)
	}

	if *outputGIF != "" {
		gifDir := filepath.Dir(*outputGIF)
		if gifDir != "." && gifDir != "" {
			if err := os.MkdirAll(gifDir, 0755); err != nil {
				log.Printf("Warning: Failed to create GIF directory: %v", err)
			}
		}

		fmt.Printf("\nSaving GIF to %s...\n", *outputGIF)
		if err := result.SaveGIF(*outputGIF); err != nil {
			log.Printf("Warning: Failed to save GIF: %v", err)
		} else {
			fmt.Printf("GIF saved to %s\n", *outputGIF)
		}
	}

	fmt.Println("\nDone!")
}
