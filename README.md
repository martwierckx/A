# ÖBB Railjet dubbeldekker – Transport Fever 3 mod

AI-gegenereerde mod (Claude, Anthropic) van de zesdelige Railjet dubbeldekker (Stadler KISS) van de ÖBB.

## Installeren

1. Kopieer de map `mod/claude_railjet_dosto` naar `…/steamapps/common/Transport Fever 3/mods/`.
2. Zet de mod aan in de modlijst van het spel.
3. In het depot staan de rijtuigen onder "ÖBB Railjet dubbeldekker".

Treinsamenstelling (486 zitplaatsen):

| Positie | Rijtuig | Bestand |
|---|---|---|
| 1 | Stuurstandrijtuig 1e klas (voor) | `rj_dosto_1_front.mdl` |
| 2 | Tussenrijtuig 2e klas met pantograaf | `rj_dosto_2_mid.mdl` |
| 3 | Multifunctioneel rijtuig | `rj_dosto_3_mz.mdl` |
| 4 | Tussenrijtuig 2e klas | `rj_dosto_4_mid.mdl` |
| 5 | Tussenrijtuig 2e klas met pantograaf | `rj_dosto_2_mid.mdl` |
| 6 | Stuurstandrijtuig 2e klas (achter, al gedraaid) | `rj_dosto_6_rear.mdl` |

## Opnieuw bouwen

De hele mod (modellen, textures, iconen, teksten) wordt door scripts gemaakt:

```
pip install numpy pillow lupa
python3 tools/build.py            # bouwt mod/claude_railjet_dosto
python3 tools/check_mod.py mod/claude_railjet_dosto
```

| Bestand | Inhoud |
|---|---|
| `tools/railjet.py` | Bakvorm: profiel, zones, ramen/deuren, cabine (neus) |
| `tools/parts.py` | Draaistellen, wielstellen, pantograaf, balgen, koppeling, interieur |
| `tools/paint.py` | Kleurstelling (gebakken uit de 3D-vorm), materiaalwaarden, glas |
| `tools/meshlib.py` | Mesh-opbouw en `.msh`/`.msh.blob`-writer |
| `tools/dds.py` | DXT1/DXT5/ATI2-encoder met mipmaps |
| `tools/build.py` | Alles samenvoegen: `.mdl`, `.mtl`, `.ani`, iconen, preview, teksten |

Het bestandsformaat (`.mdl`, `.msh` + `.msh.blob`, `.mtl`, `.ani`, `mod.json`, `strings.json`,
`_metadata/modinfo.json`) is afgeleid van een bestaande Transport Fever 3-mod.

## Nog niet in het spel getest

- Geluid: tijdelijk de standaard `train_diesel.snd`. De naam van het elektrische geluidsbestand van TF3 is nog onbekend.
- Draaistelframes staan vast aan de bak; alleen de wielstellen draaien.
- Vermogen (2 × 3000 kW), aanzetkracht en massa zijn schattingen.
- Geen bedrijfslogo's (ÖBB, railjet, Stadler).
