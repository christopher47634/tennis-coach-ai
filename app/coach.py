"""Template-based coaching advice generator.

For each joint rated 'needs_work' or 'moderate', generates specific
actionable advice. No LLM API needed — pure rule-based templates.
"""

# Advice templates keyed by (joint_name, severity_bucket).
# severity_bucket: 'needs_work' or 'moderate'
# {direction} is filled based on whether the player's angle is above or below expert mean.

ADVICE_TEMPLATES = {
    "right_elbow_flexion": {
        "needs_work": {
            "below": (
                "Your right arm is too straight at the elbow during the serve. "
                "Focus on keeping a slight bend (around {expert}°) as you accelerate "
                "toward the ball. Practice shadow swings with a towel tucked under "
                "your elbow to feel the correct flexion."
            ),
            "above": (
                "Your right elbow is too bent (measured {value}° vs ideal {expert}°). "
                "This limits your reach and power. Work on full arm extension at "
                "contact — try 'freeze frame' drills where you pause at the hit "
                "point and check arm position."
            ),
        },
        "moderate": {
            "below": (
                "Slightly straight right arm — aim for a bit more elbow bend "
                "(target ~{expert}°). A slight bend helps with pronation and spin."
            ),
            "above": (
                "Elbow flexion is a touch high. Focus on extending through the "
                "ball more at contact for extra power."
            ),
        },
    },
    "left_elbow_flexion": {
        "needs_work": {
            "below": (
                "Your left arm (toss arm) is too straight. Keep it slightly bent "
                "(~{expert}°) as you reach up for the toss — this gives better "
                "control and balance."
            ),
            "above": (
                "Your toss arm elbow is too bent at {value}°. Extend it more "
                "(target ~{expert}°) to create a stable platform and better "
                "body alignment."
            ),
        },
        "moderate": {
            "below": "Left arm could use a touch more flexion for better toss control.",
            "above": "Left arm slightly over-bent — extend it a bit more on the toss.",
        },
    },
    "right_shoulder_abduction": {
        "needs_work": {
            "below": (
                "Your right shoulder isn't abducted enough ({value}° vs {expert}°). "
                "You're keeping your arm too close to your body. Practice 'trophy "
                "position' drills: get your hitting arm up and out, elbow at "
                "shoulder height, forming an L shape."
            ),
            "above": (
                "Your shoulder is too abducted ({value}°). This can cause strain "
                "and inconsistency. Bring your elbow slightly closer to shoulder "
                "height (~{expert}°). Think 'elbow up, not out.'"
            ),
        },
        "moderate": {
            "below": "Shoulder angle is slightly low — a bit more arm elevation will improve your reach.",
            "above": "Shoulder slightly over-abducted — pull the elbow in a touch for better control.",
        },
    },
    "left_shoulder_abduction": {
        "needs_work": {
            "below": (
                "Your non-dominant shoulder needs more elevation. Raise your toss "
                "arm higher (~{expert}°) to create a better kinetic chain and "
                "improve balance."
            ),
            "above": (
                "Toss arm is too high/abducted at {value}°. Lower it slightly "
                "(~{expert}°) — over-raising can unbalance your stance."
            ),
        },
        "moderate": {
            "below": "Toss arm slightly low — raise it a bit for better balance and rotation.",
            "above": "Toss arm a touch high — ease it down slightly for stability.",
        },
    },
    "right_knee_flexion": {
        "needs_work": {
            "below": (
                "Your right leg is too straight ({value}° vs {expert}°). You need "
                "more knee bend to load power from the ground up. Practice 'sit "
                "and explode' drills — squat into your serve stance, then drive up."
            ),
            "above": (
                "Your right knee is too bent ({value}°). You're squatting too deep, "
                "which wastes energy. Keep a more athletic stance (~{expert}°) and "
                "focus on a quick, upward drive."
            ),
        },
        "moderate": {
            "below": "Slightly stiff right knee — bend a touch more for better power transfer.",
            "above": "Right knee slightly over-flexed — stand a bit taller for quicker movement.",
        },
    },
    "left_knee_flexion": {
        "needs_work": {
            "below": (
                "Left leg too stiff ({value}°). Your back leg should bend more "
                "(~{expert}°) during the loading phase. This helps with leg drive "
                "and upward momentum."
            ),
            "above": (
                "Left knee over-bent at {value}°. Straighten slightly (~{expert}°) "
                "to maintain better balance and a more efficient push-off."
            ),
        },
        "moderate": {
            "below": "Left leg could use slightly more flexion for a stronger base.",
            "above": "Left knee slightly bent too much — straighten up a bit.",
        },
    },
    "trunk_lateral_tilt": {
        "needs_work": {
            "below": (
                "Your trunk isn't tilting enough ({value}° vs {expert}°). Side-bend "
                "is critical for reaching high contact points. Practice leaning "
                "into the serve — think 'reach up and over.' Wall stretches and "
                "side-plank work help build this mobility."
            ),
            "above": (
                "Excessive trunk tilt at {value}° — you're over-leaning, which "
                "kills balance. Aim for ~{expert}°. Strengthen your core with "
                "anti-rotation exercises."
            ),
        },
        "moderate": {
            "below": "A bit more trunk side-bend would help you reach higher contact points.",
            "above": "Slightly too much trunk lean — engage your core to stay more centered.",
        },
    },
    "hip_shoulder_separation": {
        "needs_work": {
            "below": (
                "Your hip-shoulder separation is too low ({value}° vs {expert}°). "
                "You're not rotating your hips ahead of your shoulders. This is "
                "the #1 source of serve power. Practice the 'step and rotate' "
                "drill: step into the serve with your front foot while keeping "
                "your shoulders closed as long as possible."
            ),
            "above": (
                "Too much hip-shoulder separation ({value}°). Your hips are "
                "opening too far ahead, causing loss of control. Keep your front "
                "hip more closed (~{expert}°) until you're ready to explode."
            ),
        },
        "moderate": {
            "below": "Slightly less hip-shoulder rotation than ideal — work on opening your hips earlier.",
            "above": "Hips opening a bit too much — stay closed a split second longer.",
        },
    },
    "wrist_extension": {
        "needs_work": {
            "below": (
                "Your wrist isn't extended enough ({value}° vs {expert}°). A firm, "
                "extended wrist at contact is essential for power and control. "
                "Practice wrist-snap drills with a ball on the racquet face — "
                "keep the wrist locked back and snap forward at contact."
            ),
            "above": (
                "Over-extended wrist at {value}°. You may be 'cupping' the wrist "
                "too much. Aim for a neutral-to-slight extension (~{expert}°). "
                "Try serving with a continental grip to find the right wrist angle."
            ),
        },
        "moderate": {
            "below": "Wrist could extend a bit more for a stronger contact position.",
            "above": "Wrist slightly over-extended — relax the wrist slightly for cleaner contact.",
        },
    },
}


def generate_advice(joint_scores: dict) -> str:
    """
    Generate multi-line coaching advice string from joint scores.

    Args:
        joint_scores: dict from pipeline.score_angles(), keyed by joint name,
                      each with 'value', 'expert_mean', 'severity', 'rating'.

    Returns:
        Multi-line advice string with header and per-joint tips.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  TENNIS SERVE ANALYSIS — COACHING ADVICE")
    lines.append("=" * 60)

    needs_work = []
    moderate = []
    good_count = 0

    for joint_name, score in joint_scores.items():
        rating = score['rating']
        if rating == 'good':
            good_count += 1
            continue

        direction = 'below' if score['value'] < score['expert_mean'] else 'above'
        template_bucket = ADVICE_TEMPLATES.get(joint_name, {}).get(rating, {})
        template = template_bucket.get(direction, '')

        if template:
            advice = template.format(
                value=score['value'],
                expert=score['expert_mean'],
            )
        else:
            # Fallback generic advice
            if rating == 'needs_work':
                advice = (
                    f"{joint_name.replace('_', ' ').title()}: measured {score['value']}° "
                    f"(expert: {score['expert_mean']}°). Needs significant improvement."
                )
            else:
                advice = (
                    f"{joint_name.replace('_', ' ').title()}: {score['value']}° "
                    f"(expert: {score['expert_mean']}°). Close — a small adjustment will help."
                )

        if rating == 'needs_work':
            needs_work.append((joint_name, advice))
        else:
            moderate.append((joint_name, advice))

    # Summary line
    total = len(joint_scores)
    lines.append(f"\n  You scored {good_count}/{total} joints in the 'good' range.")
    lines.append("")

    if needs_work:
        lines.append("-" * 60)
        lines.append("  PRIORITY FIXES (work on these first):")
        lines.append("-" * 60)
        for name, advice in needs_work:
            pretty = name.replace('_', ' ').title()
            lines.append(f"\n  >> {pretty}")
            # Wrap advice text
            for line in _wrap(advice, 54):
                lines.append(f"     {line}")

    if moderate:
        lines.append("")
        lines.append("-" * 60)
        lines.append("  FINE-TUNING (almost there):")
        lines.append("-" * 60)
        for name, advice in moderate:
            pretty = name.replace('_', ' ').title()
            lines.append(f"\n  >> {pretty}")
            for line in _wrap(advice, 54):
                lines.append(f"     {line}")

    if good_count == total:
        lines.append("\n  Excellent! All joints are within expert range.")
        lines.append("  Keep practicing to maintain this level of technique.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _wrap(text: str, width: int = 54) -> list[str]:
    """Simple word-wrap."""
    words = text.split()
    result = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            result.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        result.append(current)
    return result
