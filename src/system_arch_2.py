#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graphviz import Digraph


def build():
    g = Digraph("HAVOC_UNIFIED", format="png")
    g.attr(rankdir="TB", fontsize="11")

    # =========================
    # DATA
    # =========================
    g.node(
        "DATA",
        "DATA\n"
        "Text examples used to learn behavior.\n"
        "Includes normal, harmful, and jailbreak prompts.\n"
        "\nInputs:\n"
        "- B_text, H_text, J_text\n"
        "\nOutputs:\n"
        "- text prompts",
        shape="box"
    )

    # =========================
    # MODULE 1
    # =========================
    g.node(
        "M1",
        "Module 1: Activation Extraction\n"
        "The model reads text and internal signals are recorded.\n"
        "Text is converted into numerical activations.\n"
        "\nInputs:\n"
        "- text prompt\n"
        "\nOutputs:\n"
        "- B_a, H_a, J_a\n"
        "- f_I (intent activation)\n"
        "- f_P (prompt activation)",
        shape="box"
    )

    # =========================
    # MODULE 2
    # =========================
    g.node(
        "M2",
        "Module 2: Concept Construction\n"
        "Safe and unsafe activations are compared.\n"
        "This creates directions pointing toward unsafe behavior.\n"
        "\nInputs:\n"
        "- B_a, H_a, J_a\n"
        "\nOutputs:\n"
        "- mu_B, mu_H, mu_J\n"
        "- v_direct, v_jb",
        shape="box"
    )

    # =========================
    # MODULE 4
    # =========================
    g.node(
        "M4",
        "Module 4: Harmful Subspace\n"
        "Unsafe activations are grouped and analyzed.\n"
        "This builds a map of where dangerous behavior appears.\n"
        "\nInputs:\n"
        "- H_a, J_a\n"
        "\nOutputs:\n"
        "- mu_HJ\n"
        "- W (PCA basis)",
        shape="box"
    )

    # =========================
    # GAME STATE
    # =========================
    g.node(
        "STATE",
        "Game State\n"
        "The current internal condition of the model.\n"
        "This state is updated every round.\n"
        "\nVariable:\n"
        "- f_t",
        shape="box"
    )

    # =========================
    # MODULE 3a
    # =========================
    g.node(
        "M3A",
        "Module 3: Risk Scoring (Before Defense)\n"
        "The current internal state is given a risk score.\n"
        "Higher values mean higher jailbreak alignment.\n"
        "\nInputs:\n"
        "- f_t\n"
        "- v_jb, v_direct\n"
        "\nOutputs:\n"
        "- J_A_t",
        shape="box"
    )

    # =========================
    # MODULE 7
    # =========================
    g.node(
        "M7",
        "Module 7: ATTACKER\n"
        "Attempts to increase risk by changing the internal state.\n"
        "Uses past feedback to adapt its strategy.\n"
        "\nInputs:\n"
        "- f_t\n"
        "- J_A_t\n"
        "- v_jb, v_direct\n"
        "- mu_HJ, W\n"
        "\nOutputs:\n"
        "- f_t_prime",
        shape="box"
    )

    # =========================
    # MODULE 5
    # =========================
    g.node(
        "M5",
        "Module 5: Concept Fuzzing\n"
        "Explores random unsafe directions.\n"
        "Used to discover new attack strategies.\n"
        "\nInputs:\n"
        "- f_t\n"
        "- v_jb, v_direct\n"
        "\nOutputs:\n"
        "- delta_f_fuzz",
        shape="box"
    )

    # =========================
    # MODULE 6
    # =========================
    g.node(
        "M6",
        "Module 6: Manifold Steering\n"
        "Follows known harmful patterns.\n"
        "Used to exploit learned weaknesses.\n"
        "\nInputs:\n"
        "- f_t\n"
        "- mu_HJ, W\n"
        "\nOutputs:\n"
        "- delta_f_steer",
        shape="box"
    )

    # =========================
    # PROPOSAL
    # =========================
    g.node(
        "PROP",
        "Attacker Proposal\n"
        "A new internal state proposed by the attacker.\n"
        "\nOutputs:\n"
        "- f_t_prime",
        shape="box"
    )

    # =========================
    # MODULE 9
    # =========================
    g.node(
        "M9",
        "Module 9: DEFENDER\n"
        "Pushes the internal state back toward safety.\n"
        "Defense strength adapts over time.\n"
        "\nInputs:\n"
        "- f_t_prime\n"
        "- v_jb\n"
        "- W\n"
        "- lambda_t\n"
        "\nOutputs:\n"
        "- f_t_plus_1\n"
        "- lambda_t_plus_1",
        shape="box"
    )

    # =========================
    # MODULE 8
    # =========================
    g.node(
        "M8",
        "Module 8: OPTIONAL SURFACE CHECK\n"
        "Generates a model response for evaluation only.\n"
        "Not used to control the defense.\n"
        "\nInputs:\n"
        "- f_t_plus_1\n"
        "\nOutputs:\n"
        "- y_t (model response)\n"
        "- g_t (safety label)",
        shape="box"
    )

    # =========================
    # MODULE 3b
    # =========================
    g.node(
        "M3B",
        "Module 3: Risk Scoring (After Defense)\n"
        "The defended state is scored again.\n"
        "This shows how much risk remains.\n"
        "\nInputs:\n"
        "- f_t_plus_1\n"
        "\nOutputs:\n"
        "- J_D_t",
        shape="box"
    )

    # =========================
    # MODULE 10 (STOPPING CONDITION)
    # =========================
    g.node(
        "M10",
        "Module 10: STABILITY CONTROLLER\n"
        "Implements the stopping condition.\n"
        "Defined in module10_stability_controller.py\n"
        "\nStop when BOTH are true:\n"
        "- Risk change ≤ tolerance over patience rounds\n"
        "- Defended risk ≤ risk_cap\n"
        "\nInputs:\n"
        "- J_A_t history\n"
        "- J_D_t history\n"
        "\nOutputs:\n"
        "- continue / stop",
        shape="box"
    )

    # =========================
    # OUTPUTS
    # =========================
    g.node(
        "OUT",
        "FINAL OUTPUTS\n"
        "Recorded results of the attacker–defender game.\n"
        "\nOutputs:\n"
        "- J_A_t curve (attacker risk)\n"
        "- J_D_t curve (defended risk)\n"
        "- lambda_t curve (defense strength)\n"
        "- g_t labels\n"
        "- stable / unstable",
        shape="box"
    )

    # =========================
    # EDGES
    # =========================
    g.edge("DATA", "M1")
    g.edge("M1", "M2")
    g.edge("M2", "M4")
    g.edge("M4", "STATE", label="start")

    g.edge("STATE", "M3A")
    g.edge("M3A", "M7")

    g.edge("M7", "M5", label="explore")
    g.edge("M7", "M6", label="exploit")

    g.edge("M5", "PROP")
    g.edge("M6", "PROP")

    g.edge("PROP", "M9")
    g.edge("M9", "M8", label="optional")
    g.edge("M9", "M3B")

    g.edge("M3B", "M10")

    g.edge(
        "M10",
        "M7",
        label="feedback\nuse f_t_plus_1"
    )

    g.edge("M10", "OUT", label="stop")

    return g


def main():
    g = build()
    out = g.render("havoc_full_system_with_stopping_condition", cleanup=True)
    print(f"[OK] Diagram written to: {out}")


if __name__ == "__main__":
    main()
