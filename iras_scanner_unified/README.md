# IRAS Scanner v2 — Unified Edition

Scanner de signaux RF automatique avec support **USB** et **RTL-TCP**.

## Modes de connexion

### Mode USB (direct)
RTL-SDR branché directement sur le PC via USB.

```bash
python iras_scanner.py
# → Choix [1 ou 2] : 1
```

**Prérequis:**
```bash
pip install pyrtlsdr numpy scipy matplotlib
```

### Mode RTL-TCP (réseau)
RTL-SDR accédé via un serveur `rtl_tcp` sur le réseau.

**Côté SDR (machine distante):**
```bash
rtl_tcp -a 0.0.0.0 -p 1234
```

**Côté scanner (ce PC):**
```bash
python iras_scanner.py
# → Choix [1 ou 2] : 2
```

Pour spécifier un autre serveur:
```bash
python iras_scanner.py --rtl-tcp-host 192.168.1.10 --rtl-tcp-port 1234
```

**Prérequis:**
```bash
pip install numpy scipy matplotlib
# (pas besoin de pyrtlsdr en mode RTL-TCP)
```

## Utilisation basique

```bash
# Mode interactif (menu)
python iras_scanner.py

# Avec paramètres personnalisés
python iras_scanner.py \
  --freq-start 88e6 \
  --freq-end 108e6 \
  --freq-step 2e6 \
  --gain 20 \
  --snr-threshold 20 \
  --n-avg 64 \
  --record-secs 3 \
  --output-dir ./dataset
```

## Paramètres CLI

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--freq-start` | 88 MHz | Fréquence de départ (Hz) |
| `--freq-end` | 108 MHz | Fréquence de fin (Hz) |
| `--freq-step` | 2 MHz | Pas de saut (Hz) |
| `--sample-rate` | 2.4 MHz | Fréquence d'échantillonnage (Hz) |
| `--gain` | 20 dB | Gain en dB (0 = AGC auto) |
| `--ppm` | 0 | Correction PPM (cristal) |
| `--snr-threshold` | 20 dB | Seuil de détection |
| `--n-avg` | 64 | Frames FFT à moyenner |
| `--record-secs` | 3 s | Durée d'enregistrement |
| `--output-dir` | `./dataset` | Dossier de sortie |
| `--offset-tuning` | - | Active offset tuning |
| `--rtl-tcp-host` | `192.168.1.12` | Hôte du serveur (RTL-TCP) |
| `--rtl-tcp-port` | `1234` | Port du serveur (RTL-TCP) |
| `--rtl-tcp-timeout` | `10 s` | Timeout socket (RTL-TCP) |

## Sorties

Pour chaque signal détecté:
- `signal_XXX.MHz_YYYYMMDD_HHMMSS.wav` — IQ stéréo (I/Q channels)
- `signal_XXX.MHz_YYYYMMDD_HHMMSS.json` — Métadonnées (fréquence, SNR, timestamp)
- `signal_XXX.MHz_YYYYMMDD_HHMMSS.png` — Graphique de diagnostic (spectrum, waterfall, constellation, etc.)

## Exemple complet

Balayage FM 88-108 MHz via RTL-TCP, PPM 5, gain auto:

```bash
python iras_scanner.py \
  --freq-start 88e6 \
  --freq-end 108e6 \
  --freq-step 2e6 \
  --gain 0 \
  --ppm 5 \
  --snr-threshold 15 \
  --rtl-tcp-host 192.168.1.20
```

Balayage USB local, gain 25 dB, offset tuning:

```bash
python iras_scanner.py \
  --freq-start 100e6 \
  --freq-end 200e6 \
  --gain 25 \
  --offset-tuning
```

## Dépannage

**Mode USB : `ModuleNotFoundError: No module named 'rtlsdr'`**
```bash
pip install pyrtlsdr
```

**RTL-TCP : `Impossible de se connecter au serveur rtl_tcp`**
- Vérifier que `rtl_tcp` tourne côté SDR
- Vérifier la plage réseau (firewall, IP)
- Tester: `nc -zv 192.168.1.12 1234`

**Pas de signaux détectés**
- Augmenter le gain (`--gain 30` au lieu de 20)
- Baisser le seuil SNR (`--snr-threshold 10`)
- Vérifier la présence de signaux avec un autre tool (ex: GQRX)

