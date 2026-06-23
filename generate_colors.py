import json

# Seasonal Palette Definitions (3 Best, 3 Avoid)
PALETTES = {
    # Light Skin (Tones 1-3)
    "light_cool": {
        "best": [
            {"name": "Sapphire Blue", "hex": "#0F52BA"},
            {"name": "Icy Pink", "hex": "#F1D4E5"},
            {"name": "Ruby Red", "hex": "#9B111E"}
        ],
        "avoid": [
            {"name": "Mustard Yellow", "hex": "#FFDB58"},
            {"name": "Warm Orange", "hex": "#FFA500"},
            {"name": "Olive Green", "hex": "#708238"}
        ]
    },
    "light_warm": {
        "best": [
            {"name": "Warm Peach", "hex": "#FFCBA4"},
            {"name": "Aqua Blue", "hex": "#00FFFF"},
            {"name": "Golden Yellow", "hex": "#FFDF00"}
        ],
        "avoid": [
            {"name": "Icy Blue", "hex": "#A9D0F5"},
            {"name": "Stark Black", "hex": "#000000"},
            {"name": "Bright Fuchsia", "hex": "#FF00FF"}
        ]
    },
    "light_neutral": {
        "best": [
            {"name": "Dusty Rose", "hex": "#DCAE96"},
            {"name": "Soft Teal", "hex": "#4A90E2"},
            {"name": "Muted Taupe", "hex": "#B38B6D"}
        ],
        "avoid": [
            {"name": "Neon Green", "hex": "#39FF14"},
            {"name": "Pure Orange", "hex": "#FF7F00"},
            {"name": "Stark White", "hex": "#FFFFFF"}
        ]
    },

    # Medium Skin (Tones 4-7)
    "medium_cool": {
        "best": [
            {"name": "Deep Ruby", "hex": "#843F5B"},
            {"name": "Stark Navy", "hex": "#000080"},
            {"name": "Emerald Green", "hex": "#50C878"}
        ],
        "avoid": [
            {"name": "Mustard Yellow", "hex": "#FFDB58"},
            {"name": "Burnt Orange", "hex": "#CC5500"},
            {"name": "Earthy Olive", "hex": "#556B2F"}
        ]
    },
    "medium_warm": {
        "best": [
            {"name": "Terracotta", "hex": "#E2725B"},
            {"name": "Rich Olive", "hex": "#4B5320"},
            {"name": "Warm Bronze", "hex": "#CD7F32"}
        ],
        "avoid": [
            {"name": "Frosty Pink", "hex": "#F8C8DC"},
            {"name": "Neon Blue", "hex": "#1F51FF"},
            {"name": "Stark White", "hex": "#FFFFFF"}
        ]
    },
    "medium_neutral": {
        "best": [
            {"name": "Deep Plum", "hex": "#673147"},
            {"name": "Slate Blue", "hex": "#516B84"},
            {"name": "Jade Green", "hex": "#00A86B"}
        ],
        "avoid": [
            {"name": "Neon Pink", "hex": "#FF6EC7"},
            {"name": "Icy Mint", "hex": "#B4F8C8"},
            {"name": "Pale Yellow", "hex": "#FFFF99"}
        ]
    },

    # Dark Skin (Tones 8-10)
    "dark_cool": {
        "best": [
            {"name": "Royal Blue", "hex": "#4169E1"},
            {"name": "Stark White", "hex": "#FFFFFF"},
            {"name": "Vivid Fuchsia", "hex": "#FF00FF"}
        ],
        "avoid": [
            {"name": "Dusty Pastel Pink", "hex": "#EBC8C1"},
            {"name": "Muddy Brown", "hex": "#654321"},
            {"name": "Pale Mustard", "hex": "#EADD80"}
        ]
    },
    "dark_warm": {
        "best": [
            {"name": "Rich Gold", "hex": "#D4AF37"},
            {"name": "Burnt Orange", "hex": "#CC5500"},
            {"name": "Deep Copper", "hex": "#B87333"}
        ],
        "avoid": [
            {"name": "Icy Silver", "hex": "#C0C0C0"},
            {"name": "Pale Cool Gray", "hex": "#D3D3D3"},
            {"name": "Frosty Lavender", "hex": "#E6E6FA"}
        ]
    },
    "dark_neutral": {
        "best": [
            {"name": "Deep Teal", "hex": "#008080"},
            {"name": "True Red", "hex": "#FF0000"},
            {"name": "Eggplant Purple", "hex": "#614051"}
        ],
        "avoid": [
            {"name": "Muddy Pastel Green", "hex": "#C5E384"},
            {"name": "Neon Yellow", "hex": "#FFFF33"},
            {"name": "Ashy Taupe", "hex": "#B0A8B9"}
        ]
    }
}

# Mapping Monk Tones to Palettes
tone_map = {}

for tone in range(1, 11):
    category = ""
    if tone <= 3:
        category = "light"
    elif tone <= 7:
        category = "medium"
    else:
        category = "dark"
    
    tone_map[f"tone_{tone}"] = {
        "cool": PALETTES[f"{category}_cool"],
        "warm": PALETTES[f"{category}_warm"],
        "neutral": PALETTES[f"{category}_neutral"],
    }

with open("monk_skin_tone_color_recommendations.json", "w", encoding="utf-8") as f:
    json.dump(tone_map, f, indent=2)

print("Successfully regenerated monk_skin_tone_color_recommendations.json")
