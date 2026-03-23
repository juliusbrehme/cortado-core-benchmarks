from typing import List, Dict, Set
import numpy as np
from tqdm import tqdm
from cortado_core.utils.split_graph import (
    Group,
    SequenceGroup,
    ParallelGroup,
    WildcardGroup, 
    LeafGroup, 
    AnythingGroup, 
    OptionalGroup,
    ChoiceGroup
)

class QueryMiner:

    def __init__(self, variants: List[SequenceGroup], seed=42, cut_probability: float = 0.2, mutation_probabilities: Dict[str, float] = None, disable_random_walk: bool = False):
        self.variants = variants
        self.rng = np.random.default_rng(seed)
        self.queries = []
        self.available_activities = self.available_activities_from_variants(variants)

        self.disable_random_walk = disable_random_walk
        self.cut_probability = cut_probability
        default_mutation_probabilities = {
            "make_wildcard": 0.25,
            "make_anything": 0.25,
            "make_optional": 0.25,
            "make_choice": 0.25,
            "make_no_match": 0.0,
        }
        self.mutation_probabilities = mutation_probabilities or default_mutation_probabilities

    def activities_from_variant(self, variant: Group) -> Set[str]:
        if isinstance(variant, LeafGroup):
            return {variant[0]}
        
        activities = set()
        for element in variant:
            activities.update(self.activities_from_variant(element))
        return activities

    def available_activities_from_variants(self, variants: List[SequenceGroup]) -> Set[str]:
        activities = set()
        for variant in variants:
            activities.update(self.activities_from_variant(variant))
        return activities

    def pick_random_variant(self):
        idx = self.rng.choice(range(len(self.variants)))
        return self.variants[idx]
    
    def cut_left(self, variant: SequenceGroup):
        cut_index = self.rng.integers(0, variant.list_length())
        del variant[:cut_index]

    def cut_right(self, variant: SequenceGroup):
        cut_index = self.rng.integers(0, variant.list_length())
        del variant[cut_index + 1 :]
    
    def make_wildcard(self, variant: SequenceGroup) -> bool:
        replace_index = self.rng.integers(0, variant.list_length())
        if not isinstance(variant[replace_index], LeafGroup):
            return False
 
        variant[replace_index] = WildcardGroup()
        return True
    
    def make_anything(self, variant: SequenceGroup) -> bool:
        if variant.list_length() < 2:
            return False
        
        replace_index = self.rng.integers(0, variant.list_length() - 1)

        variant[replace_index] = AnythingGroup()
        del variant[replace_index + 1]
        return True

    def make_optional(self, variant: SequenceGroup) -> bool:
        if variant.list_length() < 1:
            return False
        
        replace_index = self.rng.integers(0, variant.list_length())
        if isinstance(variant[replace_index], OptionalGroup):
            return False

        variant[replace_index] = OptionalGroup([variant[replace_index]])
        return True
    
    def make_choice(self, variant: SequenceGroup) -> bool:
        if variant.list_length() < 1:
            return False
        
        trials = 0
        while trials < 5:
            trials += 1
            replace_index = self.rng.integers(0, variant.list_length())
            variant_to_modify = variant[replace_index]

            if not isinstance(variant_to_modify, LeafGroup):
                continue
        
            current_activity = variant_to_modify[0]
            alternative_activities = list(self.available_activities - {current_activity})
            if not alternative_activities:
                return False
            
            new_activity = self.rng.choice(alternative_activities)

            variant[replace_index] = ChoiceGroup([LeafGroup([current_activity]), LeafGroup([new_activity])])
            return True
        return False

    def make_no_match(self, variant: SequenceGroup) -> bool:
        replace_index = self.rng.integers(0, variant.list_length())
        variant[replace_index] = LeafGroup(["NON_EXISTENT_ACTIVITY"])
        return True

    def mutate_query(self, variant: SequenceGroup | ParallelGroup, depth: int = 0):
        mutation_attempts = 0
        while mutation_attempts < 5:
            mutation_index = self.rng.integers(0, variant.list_length())
            element = variant[mutation_index]
            etype = type(variant[mutation_index])

            # 50% chance to mutate sub-group
            if not self.disable_random_walk and etype in (SequenceGroup, ParallelGroup) and self.rng.random() < 0.5:
                subgroup = etype([el for el in element])  # copy
                self.mutate_query(subgroup, depth=depth+1)
                variant[mutation_index] = subgroup
                return
            
            # chance to cut tree
            if depth == 0 and self.rng.random() <= self.cut_probability:
                cut_function = self.rng.choice([self.cut_left, self.cut_right])
                cut_function(variant)
                return
            
            # Try applying a mutation
            mutation_functions = [
                self.make_wildcard,
                self.make_anything,
                self.make_optional,
                self.make_choice,
                self.make_no_match,
            ]

            mutation_probabilities = [
                self.mutation_probabilities["make_wildcard"],
                self.mutation_probabilities["make_anything"],
                self.mutation_probabilities["make_optional"],
                self.mutation_probabilities["make_choice"],
                self.mutation_probabilities["make_no_match"],
            ]

            mutation_function = self.rng.choice(mutation_functions, p=mutation_probabilities)
            if mutation_function(variant):
                break
            mutation_attempts += 1

    def compute_query_complexity(self, query: SequenceGroup | ParallelGroup) -> Dict[str, int]:
        stats = {
            "elements": 0,
            "wildcards": 0,
            "anythings": 0,
            "optionals": 0,
            "parallels": 0,
            "choices": 0,
            "depth": 0,
        }

        def merge_stats(s1, s2):
            return {
                "elements": s1["elements"] + s2["elements"],
                "wildcards": s1["wildcards"] + s2["wildcards"],
                "anythings": s1["anythings"] + s2["anythings"],
                "optionals": s1["optionals"] + s2["optionals"],
                "parallels": s1["parallels"] + s2["parallels"],
                "choices": s1["choices"] + s2["choices"],
                "depth": max(s1["depth"], s2["depth"]),
            }

        for el in query:
            etype = type(el)
            stats["elements"] += 1

            if etype == WildcardGroup:
                stats["wildcards"] += 1

            elif etype == AnythingGroup:
                stats["anythings"] += 1

            elif etype == OptionalGroup:
                stats["optionals"] += 1
                sub_stats = self.compute_query_complexity(el)
                sub_stats["depth"] += 1
                stats = merge_stats(stats, sub_stats)

            elif etype == ParallelGroup:
                stats["parallels"] += 1
                sub_stats = self.compute_query_complexity(el)
                sub_stats["depth"] += 1
                stats = merge_stats(stats, sub_stats)

            elif etype == SequenceGroup:
                sub_stats = self.compute_query_complexity(el)
                sub_stats["depth"] += 1
                stats = merge_stats(stats, sub_stats)

            elif etype == ChoiceGroup:
                stats["choices"] += 1
                sub_stats = self.compute_query_complexity(el)
                sub_stats["depth"] += 1
                stats = merge_stats(stats, sub_stats)

        return stats

    def non_matching_query(self, query: Group) -> bool:
        """ Recursively checks whether the query contains a non-matching activity. """
        if isinstance(query, LeafGroup):
            return query[0] == "NON_EXISTENT_ACTIVITY"
        
        for element in query:
            if self.non_matching_query(element):
                return True
        return False

    def generate_queries(self, num_queries: int):
        self.queries.clear()
        for _ in tqdm(range(num_queries), total=num_queries, desc="Generating Queries"):
            mutate_existing = False
            if len(self.queries) > 3 and self.rng.random() < 0.8:
                mutate_existing = True

            if mutate_existing:
                while True:
                    idx = self.rng.integers(0, len(self.queries))
                    variant = self.queries[idx]
                    if variant.list_length() > 2:
                        break
            else:
                variant = self.pick_random_variant()

            query = SequenceGroup([elem for elem in variant])  #  copy
            self.mutate_query(query)
            self.queries.append(query)

        # Print percentage of non-matching queries
        # non_matching_count = sum(1 for q in self.queries if self.non_matching_query(q))
        # non_matching_percentage = (non_matching_count / len(self.queries)) * 100
        # print(f"  Generated {non_matching_count} non-matching queries ({non_matching_percentage:.2f}%)")

        return self.queries
