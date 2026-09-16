"""Script d'évaluation et de démonstration de scénarios d'interaction pour le Chatbot RAG.

Ce script teste la robustesse du chatbot sur plusieurs scénarios représentatifs :
1. Recherche thématique ciblée (Atelier numérique / informatique débutant)
2. Sortie en famille avec enfants (Nature, observation, chauve-souris)
3. Événement festif ou musical (Concert, jazz, festival)
4. Recherche géographique précise (Événements à Lille ou à Crépy-en-Valois)
5. Cas limite / Hors périmètre (Question hors sujet pour vérifier l'anti-hallucination)
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from chatbot import EventRAGChatbot, get_project_root

# Configuration de l'encodage de la console sur Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "1_thematic_digital",
        "nom": "Recherche thématique ciblée (Atelier numérique)",
        "question": "Bonjour, je cherche un atelier pour m'initier au numérique et aux bases de l'informatique.",
        "description": "Vérifie la capacité à recommander des formations ou ateliers d'initiation pertinents.",
        "reponse_reference_humaine": (
            "Pour vous initier au numérique et acquérir les bases de l'informatique en 2026 dans les Hauts-de-France, "
            "plusieurs ateliers pratiques et accessibles aux débutants sont programmés :\n\n"
            "1. **Les RDV numériques du vendredi – Vers l’autonomie à Vervins** :\n"
            "- **Lieu** : Tac Tic Animation, Vervins (Aisne).\n"
            "- **Période** : Les vendredis du 22 mai au 24 juillet 2026.\n"
            "- **Programme** : Ateliers pratiques en petit groupe pour apprendre pas à pas à naviguer sur Internet, "
            "gérer ses emails et utiliser les outils informatiques du quotidien en toute autonomie.\n"
            "- **Tarif** : 5 € par séance (+ adhésion annuelle).\n\n"
            "2. **ATELIERS NUMÉRIQUES à Grande-Synthe** :\n"
            "- **Lieu** : Maison de quartier Saint-Jacques, Grande-Synthe.\n"
            "- **Dates** : Séances organisées en septembre 2026 (notamment les mercredis 16 et 23 septembre 2026).\n"
            "- **Programme** : Cycle d'initiation gratuit sans prérequis pour découvrir l'ordinateur, créer une adresse mail "
            "et naviguer en toute sécurité.\n"
            "- **Conditions** : Gratuit sur inscription auprès de la maison de quartier.\n\n"
            "3. **Café Numérique à Saint-Quentin** :\n"
            "- **Programme** : Rencontres conviviales pour apprendre à démêler l'info et maîtriser les outils numériques simples.\n\n"
            "Ces ateliers conviviaux sont idéaux pour débuter à votre rythme avec l'accompagnement d'animateurs spécialisés."
        ),
        "criteres_cles": [
            {"label": "Atelier de Vervins (RDV numériques / autonomie)", "keywords": ["vervins", "rdv numérique", "rdv numerique", "autonomie", "tac tic"]},
            {"label": "Atelier de Grande-Synthe (création mail / ordinateur)", "keywords": ["grande-synthe", "ateliers numeriques", "ateliers numériques", "saint-jacques"]},
            {"label": "Thématique initiation / débutant informatique", "keywords": ["initiation", "débutant", "debutant", "numérique", "numerique", "informatique"]},
            {"label": "Mention de l'année 2026", "keywords": ["2026"]},
            {"label": "Conditions d'accès et tarifs (gratuit / 5€)", "keywords": ["gratuit", "5€", "5 €", "tarif", "inscription", "adhésion", "adhesion", "libre"]},
        ],
    },
    {
        "id": "2_family_nature",
        "nom": "Recommandation familiale (Sortie nature avec enfants)",
        "question": "Quelles activités ludiques ou sorties nature sont adaptées aux familles avec des enfants ?",
        "description": "Vérifie le ciblage par public (enfants/familles) et thématique nature.",
        "reponse_reference_humaine": (
            "Pour une sortie nature et ludique en famille avec des enfants en 2026 dans les Hauts-de-France, "
            "voici les événements recommandés dans la région :\n\n"
            "1. **Mercredi des tout-petits à Don (pour les 1-3 ans)** :\n"
            "- **Lieu** : Espace Découverte Nature et Oiseaux (DON), 3-13 rue Abbé Dubus, Don (Nord).\n"
            "- **Date** : Mercredi 9 septembre 2026 à 10h30.\n"
            "- **Activités** : Balade sensorielle et ateliers d'éveil à la nature animés par un guide spécialisé, "
            "à la découverte des oiseaux et de la petite faune locale.\n"
            "- **Accès** : Gratuit sur inscription préalable (séances en petits groupes).\n\n"
            "2. **Balade à vélo familiale à Compiègne** :\n"
            "- **Lieu** : Départ Place de l'Hôtel de Ville, Compiègne (Oise).\n"
            "- **Date** : Samedi 25 avril 2026 à 14h00.\n"
            "- **Activités** : Randonnée cycliste familiale très accessible et sécurisée le long des berges de l'Oise, "
            "avec des parcours adaptés et des arrêts ludiques pour les enfants.\n\n"
            "3. **La p'tite famille au musée à Bailleul (dès 4 ans)** :\n"
            "- **Lieu** : Musée Benoît-De-Puydt, 24 rue du Musée, Bailleul (Nord).\n"
            "- **Date** : Mercredi 26 août 2026 à 11h00.\n"
            "- **Activités** : Visite interactive conçue comme un jeu de piste avec énigmes et ateliers manuels en famille.\n\n"
            "4. **Fête du moulin à Fontaine-lès-Vervins** :\n"
            "- **Lieu** : Médiathèque Henry Duflot, Fontaine-lès-Vervins (Aisne).\n"
            "- **Date** : Dimanche 28 juin 2026 à 10h00.\n"
            "- **Activités** : Journée festive gratuite avec découverte des moulins, contes et animations artisanales en plein air."
        ),
        "criteres_cles": [
            {"label": "Événement de Don (Mercredi des tout-petits / nature)", "keywords": ["don", "tout-petits", "tout petits", "oiseau", "oiseaux"]},
            {"label": "Événement de Compiègne ou Bailleul (Balade vélo / musée)", "keywords": ["compiègne", "compiegne", "vélo", "velo", "bailleul", "musée", "musee"]},
            {"label": "Ciblage familial et enfants", "keywords": ["famille", "familles", "enfant", "enfants", "1 à 3 ans", "1-3 ans", "4 ans"]},
            {"label": "Mention de l'année 2026", "keywords": ["2026"]},
            {"label": "Conditions d'accès (gratuit / inscription)", "keywords": ["gratuit", "inscription", "accès", "acces", "libre"]},
        ],
    },
    {
        "id": "3_music_festival",
        "nom": "Recherche musicale / festive (Concert & Jazz)",
        "question": "Je voudrais assister à des concerts de jazz ou des festivals de musique en plein air.",
        "description": "Vérifie la pertinence des suggestions musicales et des conditions en plein air.",
        "reponse_reference_humaine": (
            "Pour profiter de concerts de jazz et de festivals musicaux en plein air en 2026 dans les Hauts-de-France, "
            "plusieurs événements majeurs sont au programme :\n\n"
            "1. **Morty Jazz Festival à Mortefontaine (Oise)** :\n"
            "- **Lieu** : Jardin de la Mairie et cœur de village, Mortefontaine.\n"
            "- **Dates** : Vendredi 24 et samedi 25 juillet 2026.\n"
            "- **Ambiance** : Festival de jazz convivial et entièrement gratuit en plein air, rassemblant des artistes "
            "reconnus et des musiciens émergents, complété par un stage d'initiation au jazz et des scènes sur les marchés.\n\n"
            "2. **Rencontres Musicales de Cambrai** :\n"
            "- **Lieu** : Divers lieux patrimoniaux et scènes de plein air à Cambrai.\n"
            "- **Période** : Du 23 mai au 11 juillet 2026.\n"
            "- **Programme** : Grand festival éclectique mettant à l'honneur le jazz, les musiques du monde et les musiques actuelles "
            "dans un esprit festif et populaire.\n\n"
            "3. **Concert Musiques Actuelles & Jazz à Villeneuve-d'Ascq** :\n"
            "- **Lieu** : École Municipale de Musique et de Danse, Villeneuve-d'Ascq (Nord).\n"
            "- **Date** : Lundi 8 juin 2026.\n"
            "- **Programme** : Soirée live dédiée aux formations jazz et aux musiques improvisées.\n\n"
            "4. **Concert Gospel Live en plein air à Lille** :\n"
            "- **Date** : Dimanche 21 juin 2026 lors de la Fête de la Musique à Lille, concert festif gratuit en plein air."
        ),
        "criteres_cles": [
            {"label": "Festival Morty Jazz à Mortefontaine", "keywords": ["morty", "mortefontaine", "jardin de la mairie"]},
            {"label": "Rencontres de Cambrai ou Villeneuve-d'Ascq", "keywords": ["cambrai", "rencontres musicales", "villeneuve", "villeneuve-d'ascq"]},
            {"label": "Thématique jazz et ambiance plein air", "keywords": ["jazz", "plein air", "festival", "concert"]},
            {"label": "Mention de l'année 2026", "keywords": ["2026"]},
            {"label": "Gratuité ou conditions d'accès", "keywords": ["gratuit", "libre", "stage"]},
        ],
    },
    {
        "id": "4_geographic_target",
        "nom": "Recherche géographique spécifique (Lille & Oise)",
        "question": "Quels événements ou spectacles sont programmés dans la métropole lilloise ou dans l'Oise ?",
        "description": "Vérifie la prise en compte des critères géographiques dans les métadonnées.",
        "reponse_reference_humaine": (
            "Dans la métropole lilloise et dans le département de l'Oise, plusieurs événements culturels, "
            "patrimoniaux et artistiques sont programmés en 2026 :\n\n"
            "1. **Dans la Métropole lilloise (Lille & environs)** :\n"
            "- **La Nuit des cathédrales 2026 à Lille** : À la Cathédrale Notre-Dame de la Treille, "
            "un grand rendez-vous nocturne gratuit associant illuminations, concerts, spectacles vivants et visites du patrimoine lillois.\n"
            "- **Événements professionnels et forums à Lille et Haubourdin** : Rendez-vous 'Opportunités Emploi' et 'La place de l'emploi' "
            "pour les projets professionnels et formations dans la métropole en septembre 2026.\n\n"
            "2. **Dans le département de l'Oise** :\n"
            "- **Balade à vélo familiale à Compiègne** : Randonnée nature et découverte patrimoniale au départ de la Place "
            "de l'Hôtel de Ville de Compiègne en avril 2026.\n"
            "- **Morty Jazz Festival à Mortefontaine** : Deux jours de concerts de jazz en plein air dans le sud de l'Oise fin juillet 2026.\n"
            "- **Atelier d'aide à la réparation à Nogent-sur-Oise** : Ateliers participatifs et solidaires de mécanique vélo et bricolage.\n\n"
            "3. **À proximité immédiate (Laon)** :\n"
            "- **Portes ouvertes de la Maison des Arts et Loisirs de Laon** : Week-end pluridisciplinaire les 19 et 20 septembre 2026 "
            "(théâtre, cirque, danse, musique et expositions d'art contemporain)."
        ),
        "criteres_cles": [
            {"label": "Événements métropole lilloise (Lille / Cathédrale / Emploi)", "keywords": ["lille", "cathédrale", "cathedrale", "treille", "emploi", "haubourdin"]},
            {"label": "Événements Oise ou limitrophe (Compiègne / Mortefontaine / Laon)", "keywords": ["oise", "compiègne", "compiegne", "mortefontaine", "laon", "arts et loisirs"]},
            {"label": "Mention de l'année 2026", "keywords": ["2026"]},
            {"label": "Types d'événements culturels cités", "keywords": ["spectacle", "spectacles", "culturel", "concert", "visite", "exposition", "portes ouvertes"]},
        ],
    },
    {
        "id": "5_out_of_scope",
        "nom": "Cas limite / Hors périmètre (Anti-hallucination)",
        "question": "Peux-tu m'expliquer comment réparer un moteur de voiture à Marseille ?",
        "description": "Vérifie que le bot refuse poliment sans inventer d'événements et rappelle son périmètre.",
        "reponse_reference_humaine": (
            "Je suis désolé, mais en tant que conseiller spécialisé dans les sorties et événements culturels publics de l'année 2026 "
            "dans la région Hauts-de-France, cette demande est en dehors de mon périmètre d'action.\n\n"
            "Je ne peux pas vous fournir d'instructions de mécanique automobile ni d'adresses à Marseille. "
            "Pour réparer un moteur de voiture, je vous conseille de vous rapprocher d'un garagiste certifié, d'un centre de formation "
            "automobile ou d'un manuel technique spécialisé.\n\n"
            "En revanche, si vous recherchez des événements ou ateliers participatifs de bricolage et de réparation dans les Hauts-de-France "
            "en 2026 (comme les Ateliers d'aide à la réparation de vélos à Nogent-sur-Oise ou des Repair Cafés régionaux), "
            "je me ferai un plaisir de vous renseigner !"
        ),
        "criteres_cles": [
            {"label": "Refus poli explicite", "keywords": ["désolé", "desole", "excuse", "regrette", "ne peux pas", "hors de mon périmètre", "hors périmètre"]},
            {"label": "Rappel du périmètre régional (Hauts-de-France / 2026)", "keywords": ["hauts-de-france", "2026"]},
            {"label": "Mention de la mécanique automobile / Marseille", "keywords": ["marseille", "moteur", "voiture", "mécanique", "mecanique"]},
            {"label": "Alternative constructive régionale (ateliers réparation / vélo)", "keywords": ["atelier", "ateliers", "réparation", "reparation", "bricolage", "nogent", "vélo", "velo"]},
        ],
    },
]


def evaluate_factual_criteria(
    answer: str, criteres_cles: List[Dict[str, Any]]
) -> Tuple[float, List[Dict[str, Any]]]:
    """Évalue la présence des informations clés (Exact Match factuel) par rapport aux critères annotés."""
    ans_lower = answer.lower()
    details = []
    valid_count = 0

    for crit in criteres_cles:
        label = crit["label"]
        keywords = crit["keywords"]
        matched_kw = None
        for kw in keywords:
            if kw.lower() in ans_lower:
                matched_kw = kw
                break

        is_valid = matched_kw is not None
        if is_valid:
            valid_count += 1

        details.append({
            "label": label,
            "valide": is_valid,
            "matched_keyword": matched_kw,
        })

    total = len(criteres_cles)
    score_pct = (valid_count / total * 100.0) if total > 0 else 100.0
    return round(score_pct, 1), details


def compute_token_f1(reference: str, prediction: str) -> float:
    """Calcule le score de similarité textuelle (F1 au niveau des tokens) entre la référence humaine et l'IA."""
    stop_words = {
        "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "à", "a",
        "au", "aux", "ce", "ces", "pour", "dans", "par", "sur", "qui", "que",
        "est", "sont", "avec", "d", "l", "s", "se", "qu", "ont", "vous", "votre",
        "vos", "nous", "notre", "nos", "il", "elle", "ils", "elles", "mais", "ou",
    }
    ref_tokens = [t for t in re.findall(r"\w+", reference.lower()) if t not in stop_words and len(t) > 1]
    pred_tokens = [t for t in re.findall(r"\w+", prediction.lower()) if t not in stop_words and len(t) > 1]

    if not ref_tokens or not pred_tokens:
        return 0.0

    set_ref = set(ref_tokens)
    set_pred = set(pred_tokens)
    common = set_ref & set_pred
    if not common:
        return 0.0

    precision = len(common) / len(set_pred)
    recall = len(common) / len(set_ref)
    f1 = 2 * (precision * recall) / (precision + recall) * 100.0
    return round(f1, 1)


def classify_evaluation(score_exact_match: float, token_f1: float) -> Tuple[str, str, str]:
    """Attribue une classification qualitative : Correcte / Partiellement correcte / Incorrecte."""
    if score_exact_match >= 80.0:
        return (
            "Correcte",
            "badge-success",
            "La réponse contient toutes les informations attendues et respecte fidèlement la référence annotée par l'humain.",
        )
    elif score_exact_match >= 50.0:
        return (
            "Partiellement correcte",
            "badge-warning",
            "La réponse est globalement pertinente mais omet certains détails ou critères clés attendus.",
        )
    else:
        return (
            "Incorrecte",
            "badge-danger",
            "La réponse est insuffisante, hors sujet ou ne contient pas les informations factuelles requises.",
        )


def format_markdown_to_html(text: str) -> str:
    """Convertit du texte Markdown simple en HTML sécurisé."""
    safe = html.escape(text)
    # Remplacement des balises de gras **texte**
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)

    # Conversion des listes à puces et paragraphes
    lines = safe.split("\n")
    html_parts: List[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• "):
            if not in_list:
                html_parts.append('<ul style="margin: 0.5rem 0 0.75rem 1.5rem; padding: 0;">')
                in_list = True
            content = stripped[2:].strip()
            html_parts.append(f'<li style="margin-bottom: 0.35rem;">{content}</li>')
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            if stripped:
                html_parts.append(f'<p style="margin-bottom: 0.6rem;">{stripped}</p>')

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def generate_html_report(
    execution_dt: datetime,
    results: List[Dict[str, Any]],
    total_duration: float,
    init_duration: float,
    model_name: str,
) -> str:
    """Génère le document HTML autonome du rapport d'évaluation avec date d'exécution et comparaison humaine."""
    formatted_date = execution_dt.strftime("%d/%m/%Y à %H:%M:%S")
    total_scenarios = len(results)
    correct_count = sum(1 for r in results if r["classification"] == "Correcte")
    partial_count = sum(1 for r in results if r["classification"] == "Partiellement correcte")
    incorrect_count = sum(1 for r in results if r["classification"] == "Incorrecte")

    correct_rate = (correct_count / total_scenarios * 100) if total_scenarios else 0.0
    avg_exact_match = (
        sum(r["score_exact_match"] for r in results) / total_scenarios if total_scenarios else 0.0
    )
    avg_similarity = (
        sum(r["token_f1"] for r in results) / total_scenarios if total_scenarios else 0.0
    )
    avg_latency = (
        sum(r["latency"] for r in results) / total_scenarios if total_scenarios else 0.0
    )
    total_sources = sum(len(r["sources"]) for r in results)

    # Construction du tableau récapitulatif
    table_rows = []
    for r in results:
        table_rows.append(f"""
        <tr>
            <td><strong>#{r['id']}</strong></td>
            <td><a href="#scenario-{r['id']}" style="color: #1d4ed8; text-decoration: none; font-weight: 500;">{html.escape(r['nom'])}</a></td>
            <td><strong>{r['score_exact_match']:.0f}%</strong></td>
            <td>{r['token_f1']:.1f}%</td>
            <td><span class="badge {r['badge_class']}">{r['classification']}</span></td>
            <td>{r['latency']:.2f} s</td>
            <td><span class="badge badge-source">{len(r['sources'])} docs</span></td>
        </tr>
        """)
    table_html = "\n".join(table_rows)

    # Construction des fiches détaillées par scénario
    cards = []
    for r in results:
        formatted_answer = format_markdown_to_html(r["answer"])
        formatted_reference = format_markdown_to_html(r["reponse_reference"])

        # Liste des critères factuels évalués
        criteria_items = []
        for c in r.get("criteria_details", []):
            if c["valide"]:
                status_icon = "✅"
                item_cls = "valid"
                match_note = f"<span class='matched-term'>« {html.escape(c['matched_keyword'])} »</span>"
            else:
                status_icon = "❌"
                item_cls = "invalid"
                match_note = "<span class='missing-term'>(non trouvé)</span>"

            criteria_items.append(
                f"<li class='criteria-item {item_cls}'>"
                f"<span>{status_icon}</span> <strong>{html.escape(c['label'])}</strong> : {match_note}"
                f"</li>"
            )
        criteria_html = "\n".join(criteria_items)

        # Tableau des sources documentaires FAISS
        sources_rows = []
        if r["sources"]:
            for idx, s in enumerate(r["sources"], 1):
                url_html = (
                    f'<a href="{html.escape(s["url"])}" target="_blank" rel="noopener noreferrer" style="color: #2563eb; font-weight: 500;">Lien officiel ↗</a>'
                    if s.get("url")
                    else '<span style="color: #94a3b8;">Non disponible</span>'
                )
                sources_rows.append(f"""
                <tr>
                    <td style="width: 40px; text-align: center;"><strong>{idx}</strong></td>
                    <td><strong>{html.escape(s['titre'])}</strong><br><small style="color: #64748b;">{html.escape(s['description_preview'])}</small></td>
                    <td><span class="badge badge-ville">{html.escape(s['ville'])}</span></td>
                    <td style="font-family: monospace;">{s['score_distance']:.4f}</td>
                    <td>{url_html}</td>
                </tr>
                """)
            sources_table = f"""
            <table class="sources-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Événement & extrait</th>
                        <th>Ville</th>
                        <th>Distance L2</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(sources_rows)}
                </tbody>
            </table>
            """
        else:
            sources_table = '<p style="color: #64748b; font-style: italic; margin-top: 0.5rem;">Aucune source documentaire requise (requête hors-périmètre).</p>'

        cards.append(f"""
        <div class="scenario-card" id="scenario-{r['id']}">
            <div class="scenario-header">
                <div>
                    <span class="scenario-badge">Scénario {r['id']}</span>
                    <h3 class="scenario-title">{html.escape(r['nom'])}</h3>
                    <p class="scenario-desc">🎯 <strong>Objectif :</strong> {html.escape(r['description'])}</p>
                </div>
                <div class="scenario-header-actions">
                    <span class="badge {r['badge_class']}" style="font-size: 0.9rem; padding: 0.4rem 0.85rem;">
                        {r['classification']}
                    </span>
                    <span class="badge badge-latency">⏱️ {r['latency']:.2f} s</span>
                </div>
            </div>

            <div class="scenario-body">
                <!-- Question utilisateur -->
                <div class="chat-bubble user-bubble">
                    <div class="bubble-header">👤 Question posée par l'utilisateur</div>
                    <div class="bubble-content">« {html.escape(r['question'])} »</div>
                </div>

                <!-- Comparaison côte à côte Référence Humaine vs Réponse IA -->
                <div class="comparison-grid">
                    <div class="ref-bubble">
                        <div class="bubble-header">📋 Réponse annotée par l'humain (Vérité terrain / Ground Truth)</div>
                        <div class="bubble-content">{formatted_reference}</div>
                    </div>
                    <div class="bot-bubble">
                        <div class="bubble-header">🤖 Réponse générée par le Chatbot (Mistral AI + FAISS)</div>
                        <div class="bubble-content">{formatted_answer}</div>
                    </div>
                </div>

                <!-- Bloc d'évaluation de conformité -->
                <div class="eval-card">
                    <div class="eval-header">
                        <h4 style="font-size: 1rem; color: #1e293b; font-weight: 700;">
                            🎯 Évaluation de la réponse IA par rapport à la référence humaine
                        </h4>
                        <div class="eval-metrics-pills">
                            <span class="metric-pill">Exact Match factuel : <strong>{r['score_exact_match']:.0f}%</strong></span>
                            <span class="metric-pill">Similarité lexicale F1 : <strong>{r['token_f1']:.1f}%</strong></span>
                        </div>
                    </div>
                    <p class="eval-justif"><em>{html.escape(r['justification'])}</em></p>
                    <ul class="criteria-list">
                        {criteria_html}
                    </ul>
                </div>

                <!-- Sources documentaires -->
                <details class="sources-accordion">
                    <summary class="sources-summary">
                        🔍 Sources documentaires FAISS exploitées ({len(r['sources'])})
                    </summary>
                    <div style="padding-top: 0.75rem;">
                        {sources_table}
                    </div>
                </details>
            </div>
        </div>
        """)
    cards_html = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Évaluation RAG - {formatted_date}</title>
    <style>
        :root {{
            --primary: #1e40af;
            --primary-hover: #1d4ed8;
            --accent: #eff6ff;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --success: #16a34a;
            --success-bg: #dcfce7;
            --warning: #b45309;
            --warning-bg: #fef3c7;
            --danger: #b91c1c;
            --danger-bg: #fee2e2;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: var(--font-family);
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1.5rem;
        }}

        .container {{
            max-width: 1180px;
            margin: 0 auto;
        }}

        /* Header Hero */
        .header-card {{
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            color: white;
            border-radius: 14px;
            padding: 2.25rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.2);
        }}

        .header-title {{
            font-size: 1.95rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }}

        .header-subtitle {{
            font-size: 1.05rem;
            color: #dbeafe;
            margin-bottom: 1.5rem;
        }}

        .meta-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .meta-tag {{
            background: rgba(255, 255, 255, 0.18);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.88rem;
            font-weight: 500;
        }}

        .meta-tag.highlight-date {{
            background: #ffffff;
            color: #1e3a8a;
            font-weight: 700;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.35rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }}

        .kpi-label {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            font-weight: 600;
        }}

        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--primary);
            margin-top: 0.35rem;
        }}

        .kpi-subtext {{
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}

        /* Section Card */
        .section-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.75rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        }}

        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .table-responsive {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.93rem;
        }}

        th, td {{
            padding: 0.85rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background-color: var(--accent);
            color: var(--primary);
            font-weight: 600;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: #f8fafc;
        }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}

        .badge-success {{
            background: var(--success-bg);
            color: var(--success);
        }}

        .badge-warning {{
            background: var(--warning-bg);
            color: var(--warning);
        }}

        .badge-danger {{
            background: var(--danger-bg);
            color: var(--danger);
        }}

        .badge-source {{
            background: #e0f2fe;
            color: #0369a1;
        }}

        .badge-ville {{
            background: #f1f5f9;
            color: #334155;
            font-weight: 500;
        }}

        .badge-latency {{
            background: #fef3c7;
            color: #92400e;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        /* Scenario Cards */
        .scenario-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            margin-bottom: 2.25rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            overflow: hidden;
        }}

        .scenario-header {{
            background: #f8fafc;
            border-bottom: 1px solid var(--border);
            padding: 1.25rem 1.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .scenario-header-actions {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .scenario-badge {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--primary);
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}

        .scenario-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
        }}

        .scenario-desc {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        .scenario-body {{
            padding: 1.75rem;
        }}

        /* Chat Bubbles & Comparison Grid */
        .chat-bubble {{
            border-radius: 10px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1.25rem;
        }}

        .user-bubble {{
            background: #f1f5f9;
            border-left: 4px solid #64748b;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
            margin-bottom: 1.25rem;
        }}

        @media (max-width: 900px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .ref-bubble {{
            background: #f0fdf4;
            border-left: 4px solid var(--success);
            border-radius: 10px;
            padding: 1.15rem 1.35rem;
        }}

        .bot-bubble {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            border-radius: 10px;
            padding: 1.15rem 1.35rem;
        }}

        .bubble-header {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .user-bubble .bubble-header {{ color: #475569; }}
        .ref-bubble .bubble-header {{ color: #15803d; }}
        .bot-bubble .bubble-header {{ color: var(--primary); }}

        .bubble-content {{
            font-size: 0.95rem;
            color: var(--text-main);
        }}

        /* Evaluation Breakdown Card */
        .eval-card {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }}

        .eval-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .eval-metrics-pills {{
            display: flex;
            gap: 0.75rem;
        }}

        .metric-pill {{
            background: #ffffff;
            border: 1px solid var(--border);
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.82rem;
            color: #334155;
        }}

        .eval-justif {{
            color: var(--text-muted);
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }}

        .criteria-list {{
            list-style: none;
            padding: 0;
            margin: 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 0.5rem;
        }}

        .criteria-item {{
            background: #ffffff;
            border: 1px solid var(--border);
            padding: 0.45rem 0.75rem;
            border-radius: 6px;
            font-size: 0.84rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .criteria-item.valid {{
            border-left: 3px solid var(--success);
            color: #14532d;
        }}

        .criteria-item.invalid {{
            border-left: 3px solid var(--danger);
            color: #7f1d1d;
        }}

        .matched-term {{
            color: #047857;
            font-weight: 600;
        }}

        .missing-term {{
            color: #dc2626;
            font-style: italic;
        }}

        /* Sources Accordion & Tables */
        .sources-accordion {{
            background: #fafafa;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem 1.25rem;
        }}

        .sources-summary {{
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.9rem;
        }}

        .sources-table {{
            margin-top: 0.75rem;
            font-size: 0.86rem;
        }}

        .sources-table th {{
            background: #e2e8f0;
            color: #334155;
            padding: 0.5rem 0.75rem;
        }}

        .sources-table td {{
            padding: 0.5rem 0.75rem;
        }}

        footer {{
            text-align: center;
            padding: 2rem 0;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- En-tête avec Date d'Exécution bien visible -->
        <header class="header-card">
            <h1 class="header-title">🏛️ Rapport d'Évaluation - Chatbot RAG</h1>
            <p class="header-subtitle">Banc d'essai de validation qualitative et quantitative par rapport aux réponses annotées par l'humain</p>
            <div class="meta-tags">
                <span class="meta-tag highlight-date">📅 Date d'exécution : {formatted_date}</span>
                <span class="meta-tag">⏱️ Durée totale : {total_duration:.2f} s</span>
                <span class="meta-tag">🤖 Modèle : {html.escape(model_name)}</span>
                <span class="meta-tag">🗄️ VectorStore : FAISS HNSW</span>
            </div>
        </header>

        <!-- Grille des KPI -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Scénarios Évalués</div>
                <div class="kpi-value">{total_scenarios} / {total_scenarios}</div>
                <div class="kpi-subtext">5 cas représentatifs testés</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Taux de Réponses Correctes</div>
                <div class="kpi-value" style="color: var(--success);">{correct_rate:.0f}%</div>
                <div class="kpi-subtext">{correct_count} Correctes, {partial_count} Partielles, {incorrect_count} Incorrectes</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Exact Match Moyen</div>
                <div class="kpi-value">{avg_exact_match:.0f}%</div>
                <div class="kpi-subtext">Couverture des critères factuels clés</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Similarité Moyenne (F1)</div>
                <div class="kpi-value">{avg_similarity:.1f}%</div>
                <div class="kpi-subtext">Par rapport aux références humaines</div>
            </div>
        </div>

        <!-- Tableau synthétique -->
        <section class="section-card">
            <h2 class="section-title">📊 Synthèse de l'Évaluation vs Vérité Terrain</h2>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Scénario testé</th>
                            <th>Exact Match (Faits)</th>
                            <th>Similarité F1</th>
                            <th>Classification</th>
                            <th>Latence</th>
                            <th>Sources</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_html}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Fiches détaillées des scénarios -->
        <section>
            <h2 class="section-title" style="margin-bottom: 1.5rem;">📝 Analyse Détaillée par Scénario</h2>
            {cards_html}
        </section>

        <!-- Pied de page -->
        <footer>
            <p>Rapport d'évaluation généré automatiquement • Système RAG LangChain + Mistral AI + FAISS • Hauts-de-France 2026</p>
            <p style="margin-top: 0.25rem;">Horodatage de génération : {formatted_date}</p>
        </footer>
    </div>
</body>
</html>
"""


def run_evaluation(
    output_html: Optional[str | Path] = None,
    bot: Optional[EventRAGChatbot] = None,
) -> Path:
    """Exécute la série de scénarios et génère un rapport HTML complet avec évaluation humaine et date d'exécution."""
    execution_dt = datetime.now()
    t_global_start = time.perf_counter()

    print("=" * 80)
    print("🧪 ÉVALUATION DES SCÉNARIOS D'INTERACTION DU CHATBOT RAG")
    print(f"📅 Date d'exécution : {execution_dt.strftime('%d/%m/%Y à %H:%M:%S')}")
    print("=" * 80)

    # Initialisation du chatbot si non fourni
    init_duration = 0.0
    if bot is None:
        print("Chargement du modèle et de la base vectorielle...\n")
        t0 = time.perf_counter()
        bot = EventRAGChatbot(top_k=4)
        init_duration = time.perf_counter() - t0
        print(f"✅ Modèle et base initialisés en {init_duration:.2f} s.\n")

    results: List[Dict[str, Any]] = []

    for i, scen in enumerate(SCENARIOS, 1):
        print("\n" + "=" * 80)
        print(f"🔹 Scénario {i}/{len(SCENARIOS)} : {scen['nom']}")
        print(f"📝 Objectif : {scen['description']}")
        print(f"💬 Question utilisateur : « {scen['question']} »")
        print("-" * 80)

        t_start = time.perf_counter()
        res = bot.ask(scen["question"])
        elapsed = time.perf_counter() - t_start

        # Affichage des sources récupérées par FAISS
        print(f"🔍 {len(res['sources'])} sources récupérées par FAISS :")
        for j, src in enumerate(res["sources"], 1):
            titre = src["titre"]
            ville = src["ville"]
            dist = src["score_distance"]
            print(f"   [{j}] {titre} ({ville}) - Distance L2: {dist:.4f}")

        # Affichage de la réponse générée par Mistral
        print("\n🤖 Réponse générée par le chatbot :")
        print("-" * 40)
        print(res["answer"])
        print("-" * 40)
        print(f"⏱️ Temps de réponse : {elapsed:.2f} s")

        # 1. Évaluation factuelle (Exact Match sur les critères attendus par l'humain)
        score_exact_match, criteria_details = evaluate_factual_criteria(
            res["answer"], scen.get("criteres_cles", [])
        )
        # 2. Similarité lexicale/sémantique (Token F1 vs référence humaine)
        reponse_ref = scen.get("reponse_reference_humaine", "")
        token_f1 = compute_token_f1(reponse_ref, res["answer"])

        # 3. Classification qualitative (Correcte / Partiellement correcte / Incorrecte)
        classification, badge_class, justification = classify_evaluation(
            score_exact_match, token_f1
        )

        valides_nb = sum(1 for c in criteria_details if c["valide"])
        print("\n🎯 Évaluation vs Référence Humaine :")
        print(f"   • Classification : [{classification.upper()}]")
        print(f"   • Exact Match    : {score_exact_match:.1f}% ({valides_nb}/{len(criteria_details)} critères validés)")
        print(f"   • Similarité F1  : {token_f1:.1f}%")
        for c in criteria_details:
            mark = "✅" if c["valide"] else "❌"
            kw_info = f"(terme: '{c['matched_keyword']}')" if c["valide"] else "(non trouvé)"
            print(f"     {mark} {c['label']} {kw_info}")

        results.append({
            "id": scen["id"],
            "nom": scen["nom"],
            "description": scen["description"],
            "question": scen["question"],
            "reponse_reference": reponse_ref,
            "answer": res["answer"],
            "sources": res["sources"],
            "latency": elapsed,
            "score_exact_match": score_exact_match,
            "token_f1": token_f1,
            "classification": classification,
            "badge_class": badge_class,
            "justification": justification,
            "criteria_details": criteria_details,
            "status": classification,
        })

        # Pause courte entre scénarios
        time.sleep(0.5)

    total_duration = time.perf_counter() - t_global_start

    print("\n" + "=" * 80)
    print("📋 RÉCAPITULATIF DE L'ÉVALUATION DE CONFORMITÉ")
    print("=" * 80)
    print(f"{'Scénario':<40} | {'ExactMatch':<10} | {'Simil F1':<8} | {'Latence':<8} | {'Classification'}")
    print("-" * 80)
    for r in results:
        print(f"{r['nom'][:38]:<40} | {r['score_exact_match']:>6.0f}%    | {r['token_f1']:>5.1f}%  | {r['latency']:>5.2f}s  | {r['classification']}")
    print("=" * 80)

    # Détermination du chemin de sortie pour le rapport HTML
    if output_html:
        out_path = Path(output_html)
        if not out_path.is_absolute():
            out_path = get_project_root() / out_path
    else:
        out_path = get_project_root() / "rapport" / "rapport_evaluation.html"

    # Création automatique du dossier parent s'il n'existe pas
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Génération et écriture du rapport HTML
    model_name = getattr(bot, "llm_model_name", "open-mistral-nemo")
    html_content = generate_html_report(
        execution_dt=execution_dt,
        results=results,
        total_duration=total_duration,
        init_duration=init_duration,
        model_name=model_name,
    )
    out_path.write_text(html_content, encoding="utf-8")

    print(f"\n📄 Rapport HTML généré avec succès avec la date d'exécution :")
    print(f"👉 {out_path.resolve()}\n")
    print("🎉 Tous les scénarios ont été exécutés et évalués avec succès !")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation des scénarios d'interaction pour le Chatbot RAG avec génération de rapport HTML."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Chemin du fichier HTML de sortie (par défaut: rapport/rapport_evaluation.html).",
    )
    args = parser.parse_args()
    run_evaluation(output_html=args.output)


if __name__ == "__main__":
    main()
