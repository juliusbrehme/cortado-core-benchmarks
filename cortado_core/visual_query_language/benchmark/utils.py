import io
import pickle
from typing import List
from pathlib import Path
from cortado_core.utils.split_graph import SequenceGroup, Group, LeafGroup, WildcardGroup, AnythingGroup, OptionalGroup, ParallelGroup, LoopGroup, ChoiceGroup

def load_variants(filename: str) -> List[SequenceGroup]:
    with open(Path(__file__).parent / f"resources/{filename}.p", "rb") as file:
        variants = pickle.load(file)
    return variants

def serialize_group(stream: io.StringIO, group: Group):
    gtype = type(group)

    if gtype == LeafGroup:
        stream.write(f'"{group[0]}"')
    elif gtype == WildcardGroup:
        stream.write('.')
    elif gtype == AnythingGroup:
        stream.write('*')
    elif gtype == OptionalGroup:
        stream.write("(")
        serialize_group(stream, group[0])
        stream.write(")?")
    elif gtype == SequenceGroup:
        stream.write("->(")
        for i, child in enumerate(group):
            if i > 0:
                stream.write(", ")
            serialize_group(stream, child)
        stream.write(")")
    elif gtype == ParallelGroup:
        stream.write("^(")
        for i, child in enumerate(group):
            if i > 0:
                stream.write(", ")
            serialize_group(stream, child)
        stream.write(")")
    elif gtype == LoopGroup:
        stream.write(f'{{{group.min_count}}}(')
        serialize_group(stream, group[0])
        stream.write(")")

    elif gtype == ChoiceGroup:
        stream.write("X(")
        for i, child in enumerate(group):
            if i > 0:
                stream.write(", ")
            serialize_group(stream, child)
        stream.write(")")
    else:
        raise TypeError(f"Unknown group type: {gtype}")

def check_double_anything(query: SequenceGroup) -> bool:
    anything_following_count = 0
    for element in query:
        if isinstance(element, AnythingGroup):
            anything_following_count += 1
            if anything_following_count >= 2:
                return True
            continue
        if isinstance(element, OptionalGroup):
            continue
        anything_following_count = 0
    return False
