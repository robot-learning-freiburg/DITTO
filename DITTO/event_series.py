import functools
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hodetector.utils.demo import Hands
from matplotlib.lines import Line2D

sns.set()


def idx_to_color(idx: int):
    return f"C{idx}"


def parse_hand(
    hand: Optional[Hands] = None,
    fields_of_interest: List[str] = ["contact_state", "obj_touch", "grasp"],
):
    if hand is None:
        # "Empty events"?
        return {field: None for field in fields_of_interest}
    return {field: hand.get_json()[field] for field in fields_of_interest}


# TODO Type correctly?
# in_dict: Dict[A, B], entries_count: Dict[A, Set[B]]
def add_uniques(in_dict, entries_count):
    for key, value in in_dict.items():
        entries_count[key].add(value)
    return entries_count


def visualize_hands23(
    pred_results: List[List[Hands]], fig: Optional[plt.figure] = None
):
    if fig is None:
        fig, axes = plt.subplots(2, 1, dpi=400, figsize=(12, 5), sharex=True)
    else:
        axes = fig.add_subplot(2, 1)

    fields_of_interest: List[str] = ["contact_state", "obj_touch", "grasp"]
    parse_hand_local = functools.partial(
        parse_hand, fields_of_interest=fields_of_interest
    )

    T = len(pred_results)

    left_hand_data = []
    right_hand_data = []
    unique_entries_per_field = {field: set() for field in fields_of_interest}

    # Reorder the jsons
    for t, hand_lists in enumerate(pred_results):
        for hand in hand_lists:
            parsed_hand = parse_hand_local(hand)
            if hand.hand_side == "left_hand":
                left_hand_data.append(parsed_hand)
            elif hand.hand_side == "right_hand":
                right_hand_data.append(parsed_hand)
            else:
                raise NotImplementedError(f"Unexpected {hand.hand_side = }")
            unique_entries_per_field = add_uniques(
                parsed_hand, unique_entries_per_field
            )

        if len(left_hand_data) < t:
            left_hand_data.append(parse_hand_local(None))
        if len(right_hand_data) < t:
            right_hand_data.append(parse_hand_local(None))

    # Convert unique entries sets to indices mappings --> could maybe have also used my counter dict?
    entries_to_idx = {
        field: {entry: idx for idx, entry in enumerate(entries)}
        for field, entries in unique_entries_per_field.items()
    }
    # Now we convert everything to numerical values
    # x: time
    # y: field
    # color: field entry

    def convert_to_numeric(hand_data):
        hand_data_num = [[] for _ in range(len(fields_of_interest))]
        hand_data_colors = [[] for _ in range(len(fields_of_interest))]

        for t, events in enumerate(hand_data):
            for field_idx, field in enumerate(fields_of_interest):
                if events[field] is None:
                    continue

                # hand_data_num.append([t, field_idx, entries_to_idx[field][events[field]]])
                hand_data_num[field_idx].append(t)
                hand_data_colors[field_idx].append(
                    idx_to_color(entries_to_idx[field][events[field]])
                )
        return hand_data_num, hand_data_colors

    left_hand_data_num, left_hand_data_colors = convert_to_numeric(left_hand_data)
    right_hand_data_num, right_hand_data_colors = convert_to_numeric(right_hand_data)

    # axes[0].eventplot(left_hand_data_num[..., 0:2])
    axes[0].eventplot(left_hand_data_num, colors=left_hand_data_colors, linelengths=0.7)
    axes[0].set_yticks(range(len(fields_of_interest)), fields_of_interest)
    axes[0].set_title("Left Hand")

    axes[1].eventplot(
        right_hand_data_num, colors=right_hand_data_colors, linelengths=0.7
    )
    axes[1].set_yticks(range(len(fields_of_interest)), fields_of_interest)
    axes[1].set_title("Right Hand")
    
    axes[1].set_xlim(left=0, right=T)
    axes[1].set_xlabel("Timestep")

    # Create Legends
    for idx, field in enumerate(
        fields_of_interest[::-1]
    ):  # Flip because we previously arranged from bottom to top, now we do top to bottom
        custom_lines = []
        custom_strings = []
        for (
            entry_name,
            entry_idx,
        ) in entries_to_idx[field].items():
            custom_lines.append(Line2D([0], [0], color=idx_to_color(entry_idx), lw=4))
            custom_strings.append(entry_name)

        fig.legend(
            custom_lines,
            custom_strings,
            loc="lower center",
            title=field,
            bbox_to_anchor=(0.5, -(0.1 + (idx * 0.15))),
            ncol=len(custom_lines),
        )
    fig.tight_layout()

    return fig
