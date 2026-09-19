"""Définition des scénarios de test et d'évaluation pour le système RAG (Hauts-de-France 2026).

Contient les cas d'usage interactifs, objectifs, vérités terrain humaines (Ground Truth)
et indicateurs de périmètre pour l'évaluation automatisée avec Ragas.
"""

from __future__ import annotations

from typing import Any, Dict, List

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "1_thematic_digital",
        "nom": "Recherche thématique ciblée (Atelier numérique débutant)",
        "question": "Bonjour, je suis débutant en informatique et j'habite dans l'Aisne. Quels sont les horaires, le lieu exact, le programme et le tarif de l'atelier d'initiation au numérique organisé à Vervins en 2026 ?",
        "description": "Vérifie l'extraction précise et exclusive de l'atelier d'initiation au numérique à Vervins (Aisne).",
        "is_out_of_scope": False,
        "reponse_reference_humaine": (
            "À Vervins (02140), dans l'Aisne, le Centre Socioculturel Tac Tic Animation organise le cycle d'ateliers « Les RDV numériques du vendredi – Vers l’autonomie » :\n\n"
            "- **Lieu exact** : Tac Tic Animation, 9 avenue Paul Doumer, 02140 Vervins (Aisne).\n"
            "- **Dates et horaires** : Les vendredis du 22 mai au 24 juillet 2026, de 14h00 à 15h30.\n"
            "- **Programme** : Ateliers pratiques hebdomadaires en petit groupe pour gagner en autonomie dans les usages du quotidien : gestion des fichiers et des photos, prise en main du smartphone, personnalisation de son ordinateur, découverte d'outils comme Canva ou Excel, et initiation à Linux.\n"
            "- **Tarif et adhésion** : 5 € par séance (+ 2,50 € d'adhésion annuelle à l'association).\n"
            "- **Modalités d'inscription** : Renseignements et inscription par téléphone au 03 23 97 79 72 ou directement auprès de Tac Tic Animation."
        ),
    },
    {
        "id": "2_family_nature",
        "nom": "Recommandation familiale (Éveil nature tout-petits)",
        "question": "Bonjour, je cherche une activité nature gratuite pour mon enfant de 2 ans dans le département du Nord en septembre 2026. Où et quand a lieu le 'Mercredi des tout-petits', et quelles sont les modalités d'inscription ?",
        "description": "Vérifie le ciblage pour les tout-petits (1-3 ans) à Don, la gratuité et les contacts d'inscription.",
        "is_out_of_scope": False,
        "reponse_reference_humaine": (
            "L'animation nature « Mercredi des tout-petits » répond exactement à votre demande pour votre enfant de 2 ans dans le département du Nord :\n\n"
            "- **Lieu exact** : Espace Découverte Nature et Oiseaux (DON), site de la Louvière, parking 3-13 rue Abbé Dubus, 59272 Don (Nord).\n"
            "- **Date et horaire** : Mercredi 9 septembre 2026 de 10h30 à 11h30 (durée d'environ 1 heure).\n"
            "- **Public cible** : Tout-petits de 1 à 3 ans accompagnés de leurs parents.\n"
            "- **Programme** : Parcours sensoriel et découverte de la nature adaptée aux saisons avec un animateur nature de la Métropole Européenne de Lille.\n"
            "- **Modalités d'accès** : Atelier gratuit sur inscription obligatoire par email à animationnature@lillemetropole.fr ou par téléphone au 06 30 46 37 16 (jauge limitée)."
        ),
    },
    {
        "id": "3_music_festival",
        "nom": "Recherche musicale festive (Concert jazz plein air)",
        "question": "Quelles sont les dates exactes, le lieu précis et les tarifs du festival de jazz en plein air 'Morty Jazz Festival' organisé dans l'Oise à l'été 2026 ?",
        "description": "Vérifie la localisation exacte à Mortefontaine (Oise), la gratuité en plein air, les dates et le format du festival.",
        "is_out_of_scope": False,
        "reponse_reference_humaine": (
            "Le « Morty Jazz Festival » est un festival de jazz en plein air convivial et réputé dans le département de l'Oise :\n\n"
            "- **Lieu précis** : Jardin de la Mairie, 18 Rue Corot, 60128 Mortefontaine (dans le sud de l'Oise).\n"
            "- **Dates exactes** : Vendredi 24 et samedi 25 juillet 2026.\n"
            "- **Format et ambiance** : Festival de jazz en plein air privilégiant les groupes régionaux et émergents, avec concerts gratuits en centre-ville et sur les marchés, stage de jazz, jam sessions et concerts assis confortablement dans des canapés en plein air. Restauration artisanale sur place.\n"
            "- **Tarif et accès** : Entièrement gratuit et en accès libre pour les concerts en plein air (site officiel : https://mortyjazz.wordpress.com/)."
        ),
    },
    {
        "id": "4_geographic_target",
        "nom": "Recherche patrimoniale ciblée (La Nuit des cathédrales à Lille)",
        "question": "Quels sont les détails (date, lieu exact, tarif et programme) de 'La Nuit des cathédrales 2026' organisée à Lille ?",
        "description": "Vérifie la restitution précise de l'événement nocturne patrimonial à la Cathédrale de la Treille à Lille.",
        "is_out_of_scope": False,
        "reponse_reference_humaine": (
            "« La Nuit des cathédrales - 2026 » est un événement culturel et patrimonial majeur organisé à Lille :\n\n"
            "- **Lieu exact** : Cathédrale Notre-Dame de la Treille, Place Gilleson, 59000 Lille (Nord).\n"
            "- **Date et horaire** : Samedi 9 mai 2026 à partir de 18h30 (en nocturne à la tombée du soleil).\n"
            "- **Programme culturel** : Mise en lumière nocturne de la cathédrale, concerts de musique sacrée et profane, visites guidées du patrimoine et de la crypte néogothique, spectacles vivants et expositions.\n"
            "- **Tarif** : Événement en accès libre et entièrement gratuit pour tous les publics."
        ),
    },
    {
        "id": "5_out_of_scope",
        "nom": "Cas limite / Hors périmètre (Anti-hallucination & Rebond)",
        "question": "Peux-tu m'expliquer pas à pas comment démonter et réparer le moteur d'une Peugeot 206 dans un garage solidaire à Marseille ?",
        "description": "Vérifie le refus poli de la demande hors-sujet, le rappel strict du périmètre régional 2026 et la réorientation vers les ateliers locaux.",
        "is_out_of_scope": True,
        "reponse_reference_humaine": (
            "Je suis désolé, mais en tant qu'assistant spécialisé exclusivement dans les sorties et événements culturels et de loisirs en région Hauts-de-France pour l'année 2026, cette demande sort de mon périmètre d'action.\n\n"
            "Je ne peux pas vous fournir de tutoriel de mécanique automobile ni d'adresses de garages à Marseille. Pour réparer votre véhicule, je vous recommande de faire appel à un garagiste professionnel, de consulter la Revue Technique Automobile (RTA) ou de contacter un garage solidaire associatif en région Provence-Alpes-Côte d'Azur.\n\n"
            "En revanche, si vous recherchez des événements ou ateliers participatifs de bricolage et de réparation dans les Hauts-de-France en 2026 (comme les Ateliers d'aide à la réparation de vélos à Nogent-sur-Oise ou des Repair Cafés régionaux), n'hésitez pas à me demander !"
        ),
    },
]

__all__ = ["SCENARIOS"]
