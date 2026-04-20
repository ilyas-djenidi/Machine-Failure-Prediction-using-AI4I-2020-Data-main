"""
=============================================================================
RAPPORT DE DIAGNOSTIC — generate_report.py
Générateur de rapports de maintenance prédictive (Français / Algérie)
=============================================================================
Usage:
  from generate_report import generate_report
  report = generate_report(prediction_result)
  print(report["text"])
  report["save"]("reports/rapport_M01.txt")
=============================================================================
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Failure mode details (French)
# ---------------------------------------------------------------------------
FAILURE_DETAILS = {
    "TWF": {
        "nom"   : "Défaut d'usure d'outil (TWF)",
        "cause" : (
            "L'usure de l'élément mécanique a dépassé le seuil critique (≥ 200 min). "
            "Les frottements et les contraintes mécaniques augmentent, "
            "entraînant une dégradation accélérée des composants."
        ),
        "actions": [
            "1. Arrêter la machine et inspecter l'outil ou la pièce usée.",
            "2. Remplacer l'élément défaillant par une pièce conforme.",
            "3. Inspecter les roulements et le palier de l'arbre.",
            "4. Effectuer un contrôle d'alignement de l'axe moteur.",
            "5. Réaliser une analyse vibratoire après remplacement.",
        ],
        "siemens_param": "Vérifier le compteur d'heures de marche dans le bloc FC (OB1 / DB_Moteur).",
        "schneider_param": "Consulter le registre Modbus MW120 (wear counter) via EcoStruxure.",
    },
    "HDF": {
        "nom"   : "Défaut de dissipation thermique (HDF)",
        "cause" : (
            "La différence entre la température de process et la température ambiante "
            "est inférieure à 8,6 K, combinée à une vitesse de rotation inférieure à 1 380 tr/min. "
            "Le système de refroidissement est insuffisant — risque de surchauffe des enroulements."
        ),
        "actions": [
            "1. Nettoyer le ventilateur et les ailettes de refroidissement du moteur.",
            "2. Vérifier l'obstruction des grilles d'aération.",
            "3. Contrôler la température ambiante de la salle de machines.",
            "4. Vérifier le bon fonctionnement du ventilateur externe (si présent).",
            "5. Mesurer la résistance des enroulements à froid et à chaud.",
        ],
        "siemens_param": "Vérifier le bloc de protection thermique PT100 dans WinCC (TagName: MotorTemp).",
        "schneider_param": "Vérifier le relais de protection thermique TeSys D / GV3 (classe 10A).",
    },
    "PWF": {
        "nom"   : "Défaut de puissance (PWF)",
        "cause" : (
            "La puissance mécanique calculée (P = C × ω) est en dehors de la plage nominale "
            "[3 500 – 9 500 W]. Cela peut indiquer une surcharge, une sous-charge anormale, "
            "un défaut du variateur de fréquence (VFD), ou un problème d'alimentation."
        ),
        "actions": [
            "1. Vérifier la charge mécanique raccordée au moteur.",
            "2. Contrôler les paramètres du variateur (fréquence, courant limite).",
            "3. Mesurer la tension et le courant d'alimentation (trois phases).",
            "4. Vérifier les protections thermiques (relais, disjoncteur).",
            "5. Contrôler l'état des condensateurs de compensation si présents.",
        ],
        "siemens_param": "Vérifier le bloc de régulation dans le variateur Sinamics S120/G120 (r0027 = courant actuel).",
        "schneider_param": "Consulter les paramètres ATV320/ATV630 : P1.2 (courant nominal), F602 (défaut surcharge).",
    },
    "OSF": {
        "nom"   : "Défaut de surcontrainte (OSF)",
        "cause" : (
            "L'indice de surcontrainte (usure × couple) dépasse 13 000 Nm·min. "
            "Le moteur ou la transmission fonctionne au-delà de ses limites de conception, "
            "risquant une rupture mécanique ou un défaut d'arbre."
        ),
        "actions": [
            "1. Réduire immédiatement la charge ou le couple appliqué.",
            "2. Inspecter les accouplements et la transmission mécanique.",
            "3. Vérifier le dimensionnement du moteur par rapport à la charge réelle.",
            "4. Planifier une révision complète de la ligne de transmission.",
            "5. Vérifier l'état des garnitures d'étanchéité et du joint d'arbre.",
        ],
        "siemens_param": "Activer le limiteur de couple dans TIA Portal (FB_DriveControl, paramètre p1520).",
        "schneider_param": "Configurer la limite de couple dans EcoStruxure (F612 = limite couple moteur).",
    },
    "RNF": {
        "nom"   : "Défaut aléatoire (RNF)",
        "cause" : (
            "Défaillance aléatoire détectée. Ce type de défaut peut résulter d'une anomalie "
            "électrique transitoire, d'une perturbation EMI, d'un défaut de capteur, "
            "ou d'une cause externe non prévue."
        ),
        "actions": [
            "1. Inspecter le câblage et toutes les connexions électriques.",
            "2. Vérifier l'état et la calibration des capteurs et transmetteurs.",
            "3. Consulter les journaux d'alarmes SCADA/WinCC (AlarmLogging).",
            "4. Effectuer un test d'isolement (Megger test ≥ 1 MΩ).",
            "5. Vérifier la mise à la terre du moteur et du tableau électrique.",
        ],
        "siemens_param": "Consulter le journal diagnostic dans WinCC : HMI#MEE1Alg (AlarmLogging).",
        "schneider_param": "Vérifier le registre de défaut F600 dans l'ATV via PowerSuite.",
    },
}

RISK_DESCRIPTIONS = {
    "CRITIQUE" : "⛔ ARRÊT IMMÉDIAT REQUIS — Défaillance imminente dans 1 à 3 jours.",
    "URGENT"   : "🚨 INTERVENTION URGENTE — Maintenance dans les 3 à 7 jours.",
    "ATTENTION": "⚠️  ATTENTION — Planifier une maintenance dans 7 à 14 jours.",
    "SURVEILLER": "🔵 SURVEILLANCE — Augmenter la fréquence de contrôle (14–30 jours).",
    "NORMAL"   : "✅ ÉTAT NORMAL — Aucune anomalie détectée. Prochain contrôle préventif selon planning.",
}


def generate_report(prediction: dict, manufacturer: str = "Siemens") -> dict:
    """
    Generate a full French diagnostic report from a prediction dict.

    Args:
        prediction   : output of MotorFailurePredictor.predict()
        manufacturer : 'Siemens' or 'Schneider'

    Returns:
        dict with keys:
          'text'  — full printable report (str)
          'save'  — callable(path) to save to file
          'lines' — list of report lines
    """
    ts    = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
    mid   = prediction.get("machine_id", "N/A")
    prob  = prediction.get("failure_probability_pct", 0.0)
    risk  = prediction.get("risk_level", "NORMAL")
    icon  = prediction.get("risk_icon", "🟢")
    ttf   = prediction.get("time_to_failure_estimate", "—")
    modes = prediction.get("likely_failure_modes", [])
    snap  = prediction.get("sensor_snapshot", {})
    conf  = prediction.get("confidence", 0.0)
    manuf = manufacturer.upper()

    lines = []
    sep   = "=" * 68

    def L(s=""): lines.append(s)

    L(sep)
    L("  RAPPORT DE MAINTENANCE PRÉDICTIVE — MOTEUR ASYNCHRONE")
    L(f"  Système  : Prédiction de défaillance (IA / Machine Learning)")
    L(f"  Fabriquant: {manuf} (Algérie)")
    L(sep)
    L(f"  Machine ID    : {mid}")
    L(f"  Date / Heure  : {ts}")
    L(f"  Modèle IA     : XGBoost + LightGBM Ensemble (Optuna tuned)")
    L()

    L("─" * 68)
    L("  RÉSULTAT DU DIAGNOSTIC")
    L("─" * 68)
    L(f"  Probabilité de défaillance : {prob:.1f} %")
    L(f"  Niveau de risque           : {icon} {risk}")
    L(f"  Délai estimé avant panne   : {ttf}")
    L(f"  Confiance du modèle        : {conf*100:.1f} %")
    L()
    L(f"  → {RISK_DESCRIPTIONS.get(risk, '')}")
    L()

    # Sensor snapshot
    L("─" * 68)
    L("  DONNÉES CAPTEURS (snapshot)")
    L("─" * 68)
    L(f"  Température ambiante  : {snap.get('air_temp_K', 'N/A')} K  "
      f"({_k_to_c(snap.get('air_temp_K'))} °C)")
    L(f"  Température moteur    : {snap.get('motor_temp_K', 'N/A')} K  "
      f"({_k_to_c(snap.get('motor_temp_K'))} °C)")
    L(f"  Vitesse de rotation   : {snap.get('rpm', 'N/A')} tr/min")
    L(f"  Couple                : {snap.get('torque_Nm', 'N/A')} Nm")
    L(f"  Usure / Heures marche : {snap.get('wear_min', 'N/A')} min")
    L(f"  Puissance mécanique   : {snap.get('power_W', 'N/A')} W")
    if snap.get("air_temp_K") and snap.get("motor_temp_K"):
        tdiff = snap["motor_temp_K"] - snap["air_temp_K"]
        L(f"  ΔT (process − ambiant): {tdiff:.1f} K  "
          f"{'⚠️ FAIBLE' if tdiff < 8.6 else 'OK'}")
    L()

    # Failure modes
    L("─" * 68)
    L("  MODES DE DÉFAILLANCE IDENTIFIÉS")
    L("─" * 68)
    active_modes = [m for m in modes if m != "—"]
    if not active_modes:
        L("  Aucun mode de défaillance spécifique détecté.")
        L("  Surveillance générale recommandée.")
    else:
        for mode in active_modes:
            info = FAILURE_DETAILS.get(mode, {})
            L()
            L(f"  ▶ {info.get('nom', mode)}")
            L(f"    Cause : {info.get('cause', 'N/A')}")
            L()
            L("    Actions correctives :")
            for act in info.get("actions", []):
                L(f"      {act}")
            L()
            # Manufacturer-specific parameter
            if manuf == "SIEMENS":
                L(f"    Siemens : {info.get('siemens_param', '—')}")
            elif manuf == "SCHNEIDER":
                L(f"    Schneider: {info.get('schneider_param', '—')}")
    L()

    # Maintenance schedule recommendation
    L("─" * 68)
    L("  RECOMMANDATION DE PLANIFICATION")
    L("─" * 68)
    sched = _maintenance_schedule(risk)
    for s in sched:
        L(f"  {s}")
    L()

    L(sep)
    L("  Ce rapport est généré automatiquement par le système IA de")
    L("  maintenance prédictive. Il ne remplace pas le jugement d'un")
    L("  technicien qualifié. Conserver ce document dans le registre")
    L("  de maintenance de l'installation.")
    L(sep)

    text = "\n".join(lines)

    def save_fn(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[RAPPORT] Sauvegardé : {path}")

    return {"text": text, "save": save_fn, "lines": lines}


def _k_to_c(k):
    if k is None:
        return "N/A"
    return f"{k - 273.15:.1f}"


def _maintenance_schedule(risk: str) -> list[str]:
    schedules = {
        "CRITIQUE" : [
            "⛔ Arrêter la machine IMMÉDIATEMENT.",
            "→ Convoquer l'équipe de maintenance d'urgence.",
            "→ Ne pas remettre en marche avant inspection complète.",
            "→ Notifier le responsable de production.",
        ],
        "URGENT"   : [
            "🚨 Planifier une intervention dans les 48–72 heures.",
            "→ Réduire la charge de la machine si possible.",
            "→ Augmenter la fréquence de surveillance (toutes les 2 heures).",
            "→ Préparer les pièces de rechange nécessaires.",
        ],
        "ATTENTION": [
            "⚠️  Planifier une inspection dans la semaine en cours.",
            "→ Effectuer une vérification quotidienne des paramètres.",
            "→ Préparer le plan de maintenance préventive.",
            "→ Vérifier la disponibilité des pièces en stock.",
        ],
        "SURVEILLER": [
            "🔵 Surveiller les paramètres avec une fréquence accrue.",
            "→ Intégrer à la prochaine révision périodique (< 30 jours).",
            "→ Documenter l'évolution des tendances dans le GMAO.",
        ],
        "NORMAL"   : [
            "✅ Continuer selon le planning de maintenance préventive.",
            "→ Prochain contrôle selon la fiche technique du moteur.",
            "→ Archiver ce rapport dans le dossier machine.",
        ],
    }
    return schedules.get(risk, ["Consulter le responsable maintenance."])
