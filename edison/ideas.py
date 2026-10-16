"""Score-blind idea generation and provenance, separate from experiment memory."""
from collections import deque
from dataclasses import dataclass, replace
from .world import rng_for


@dataclass(frozen=True)
class Construction:
    path: tuple[str, ...] = ()
    transforms: tuple[str, ...] = ()
    repetitions: int = 1
    flip: bool = False


@dataclass(frozen=True)
class Primitives:
    nodes: tuple[str, ...]
    transforms: tuple[str, ...]


@dataclass(frozen=True)
class Proposal:
    construction: Construction
    operation: str


class Generator:
    """Finite branching; every finite composition is reachable from the root.

    No scores, experiment memory, world labels, or private parameters are inputs.
    This is a structural reference generator, not a reasoning model.
    """

    def expand(self, parent: Construction, primitives: Primitives) -> tuple[Proposal, ...]:
        proposals = [Proposal(replace(parent, path=parent.path + (node,)), "append_node")
                     for node in primitives.nodes]
        proposals.extend(Proposal(replace(parent, transforms=parent.transforms + (token,)),
                                  "append_transform") for token in primitives.transforms)
        proposals.extend((Proposal(replace(parent, repetitions=parent.repetitions + 2),
                                   "add_repetitions"),
                          Proposal(replace(parent, flip=not parent.flip), "toggle_flip")))
        # Back edges retain provenance without re-enqueuing previously known ideas.
        if parent.path:
            proposals.append(Proposal(replace(parent, path=parent.path[:-1]), "remove_node"))
        if parent.transforms:
            proposals.append(Proposal(replace(parent, transforms=parent.transforms[:-1]),
                                      "remove_transform"))
        if parent.repetitions > 1:
            proposals.append(Proposal(replace(parent, repetitions=parent.repetitions - 2),
                                      "remove_repetitions"))
        return tuple(proposals)


@dataclass(frozen=True)
class Idea:
    id: int
    construction: Construction
    discovery_depth: int


@dataclass(frozen=True)
class Edge:
    parent: int
    child: int
    operation: str


class IdeaGraph:
    def __init__(self):
        root = Idea(0, Construction(), 0)
        self.nodes = [root]
        self.edges: list[Edge] = []
        self._index = {root.construction: 0}
        self._edge_index: set[Edge] = set()
        self.duplicate_proposals = 0

    def register(self, parent: int, proposal: Proposal) -> tuple[Idea, bool]:
        origin = self.nodes[parent]
        is_new = proposal.construction not in self._index
        if is_new:
            child = Idea(len(self.nodes), proposal.construction, origin.discovery_depth + 1)
            self.nodes.append(child)
            self._index[child.construction] = child.id
        else:
            self.duplicate_proposals += 1
            child = self.nodes[self._index[proposal.construction]]
        edge = Edge(parent, child.id, proposal.operation)
        if edge not in self._edge_index:
            self.edges.append(edge)
            self._edge_index.add(edge)
        return child, is_new


class Branches:
    """FIFO gives every registered idea a turn if execution continues indefinitely.

    This guarantee uses finite branching, terminating expansion, and no pruning.
    A finite experiment budget still leaves pending ideas. Random frontier choice
    is a comparison method, without the FIFO guarantee.
    """

    def __init__(self, primitives: Primitives, method: str = "fair", seed: int = 0):
        if method not in ("fair", "random"):
            raise ValueError("Expected fair or random")
        self.primitives, self.method = primitives, method
        self.generator = Generator()
        self.graph = IdeaGraph()
        self.pending = deque([0])
        self._rng = rng_for("frontier", seed)
        self._expanded: set[int] = set()

    def update_primitives(self, primitives: Primitives):
        """New entities also grow children of previously expanded ideas.

        Only identifiers are used; no learned properties or scores enter this method.
        Existing branches cannot be removed through a catalog update.
        """
        old_nodes, old_transforms = set(self.primitives.nodes), set(self.primitives.transforms)
        if not old_nodes.issubset(primitives.nodes) or not old_transforms.issubset(primitives.transforms):
            raise ValueError("Catalog updates must be additive")
        nodes = set(primitives.nodes) - old_nodes
        transforms = set(primitives.transforms) - old_transforms
        self.primitives = primitives
        for idea_id in sorted(self._expanded):
            c = self.graph.nodes[idea_id].construction
            for proposal in self.generator.expand(c, primitives):
                if ((proposal.operation == 'append_node' and proposal.construction.path[-1] in nodes)
                        or (proposal.operation == 'append_transform'
                            and proposal.construction.transforms[-1] in transforms)):
                    self.suggest(idea_id, proposal)

    def suggest(self, parent: int, proposal: Proposal) -> tuple[Idea, bool]:
        """Register an external idea without removing any existing branch."""
        child, is_new = self.graph.register(parent, proposal)
        if is_new:
            self.pending.append(child.id)
        return child, is_new

    def next(self, preferred: int | None = None) -> Idea:
        if not self.pending:
            raise StopIteration
        if preferred is not None:
            self.pending.remove(preferred)
            idea = self.graph.nodes[preferred]
        else:
            offset = self._rng.randrange(len(self.pending)) if self.method == "random" else 0
            self.pending.rotate(-offset)
            idea = self.graph.nodes[self.pending.popleft()]
            self.pending.rotate(offset)
        for proposal in self.generator.expand(idea.construction, self.primitives):
            child, is_new = self.graph.register(idea.id, proposal)
            if is_new:
                self.pending.append(child.id)
        self._expanded.add(idea.id)
        return idea
