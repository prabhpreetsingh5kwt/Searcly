import replicate

output = replicate.run(
    "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e",
    input={
        "prompt": "An astronaut riding a rainbow unicorn"
    }
)
print(output)