"""Original three-person cohort: exact phase-preserving95 source selection."""
from collections import Counter,defaultdict
import hashlib
import math
from pathlib import Path
import numpy as np
from .support import sha,read as read_json
SCHEMA = "abcurves.tracking_control.training_windows.v1"
HISTORY_MS,FUTURE_MS,STRIDE_MS = 160,1024,64
TIE_SEED = 20260913
ARRAY_KEYS = ("xy","initial","delta","target","radius","available","valid")

def canonical_person(value):
    value = str(value).replace("participant_", "person")
    if value not in ("person1", "person2", "person3"):
        raise ValueError("Only the three declared development people are allowed")
    return value

def distribution(values):
    a = np.asarray(values, np.float64)
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), mean=float(a.mean()), **dict(zip(("min", "p50", "p80", "p95", "max"),
                map(float, np.quantile(a, [0, .5, .8, .95, 1])))))

class TrainingData:
    def __init__(self, index_path):
        self.index_path = Path(index_path).resolve()
        self.index_sha256 = sha(self.index_path)
        self.index = read_json(self.index_path)
        if self.index.get("schema") != "abcurves.tracking_control.episodes.v1":
            raise ValueError("Unsupported development index schema")
        self.rows = self.index["episodes"]
        if not self.rows or len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Empty or duplicate episode roster")
        role_path = self.index_path.parent / "roles.json"
        if not role_path.is_file() or sha(role_path) != self.index["roles_sha256"]:
            raise ValueError("Role authority hash mismatch")
        self.roles_path = role_path
        self.people = [canonical_person(r["person"]) for r in self.rows]
        scopes = {"parent": {}, "group": {}}
        for r in self.rows:
            if r["split"] not in ("train", "validation"):
                raise ValueError("Held calibration/test data is forbidden in this loader")
            for key in ("family", "parent", "group", "file", "sha256"):
                if not r.get(key):
                    raise ValueError(f"Missing episode custody field: {key}")
            if int(r["ticks"]) < 1 or not math.isfinite(float(r["source_to_common"])) or r["source_to_common"] <= 0:
                raise ValueError("Invalid episode length or native/common conversion")
            if not isinstance(r["guide"], bool):
                raise ValueError("Guide metadata must be Boolean")
            for key, seen in scopes.items():
                if r[key] in seen and seen[r[key]] != r["split"]:
                    raise ValueError(f"TRAIN/validation {key} leakage")
                seen[r[key]] = r["split"]
            path = (self.index_path.parent / r["file"]).resolve()
            if not path.is_relative_to(self.index_path.parent):
                raise ValueError("Episode file leaves the declared development directory")
        self.cache = {}
        self.windows = None
        self.scales = None
        self.receipt = None


    def episode(self, row_index):
        row_index = int(row_index)
        if row_index not in self.cache:
            row = self.rows[row_index]
            path = (self.index_path.parent / row["file"]).resolve()
            if sha(path) != row["sha256"]:
                raise ValueError("Episode data hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                ep = {k:z[k] for k in ARRAY_KEYS}
            n = int(row["ticks"])
            for key in ("xy", "delta", "target", "radius"):
                if ep[key].shape != (n, 2) or not np.isfinite(ep[key]).all():
                    raise ValueError(f"Invalid episode {key}")
            if ep["initial"].shape != (2,) or not np.isfinite(ep["initial"]).all():
                raise ValueError("Invalid initial position")
            if ep["available"].shape != (n,) or ep["available"].dtype.kind not in "iu" or ep["valid"].shape != (n,) or ep["valid"].dtype.kind != "b":
                raise ValueError("Invalid receipt/valid contract")
            if np.any(ep["radius"] <= 0) or np.any(ep["available"] > np.arange(1,n+1,dtype=np.int64)*1000):
                raise ValueError("Invalid radius or future target receipt")
            reconstructed = ep["xy"] - np.vstack((ep["initial"], ep["xy"][:-1]))
            if not np.array_equal(reconstructed, ep["delta"]):
                raise ValueError("Delta is not the exact increment of the supplied position channel")
            for a in ep.values():
                a.setflags(write=False)
            self.cache[row_index] = ep
        return self.cache[row_index]


    def build(self):
        if self.windows is not None:
            return self
        episodes, cuts, scores, outside, excess = [], [], [], [], []
        train_ticks = 0
        sums = dict(velocity=0., position=0., target_error=0., radius=0.)
        skipped = Counter()
        for i, row in enumerate(self.rows):
            ep = self.episode(i)
            n = len(ep["delta"])
            candidates = np.arange(HISTORY_MS, n-FUTURE_MS+1, STRIDE_MS, dtype=np.int64)
            bad = np.r_[0, np.cumsum(~ep["valid"], dtype=np.int64)]
            eligible = bad[candidates+FUTURE_MS] == bad[candidates-HISTORY_MS]
            skipped[row["split"]] += int(np.sum(~eligible))
            candidates = candidates[eligible]
            if row["split"] == "train":
                ok = ep["valid"]
                train_ticks += int(ok.sum())
                for key, values in (("velocity", ep["delta"]), ("position", ep["xy"]-ep["initial"]),
                                    ("target_error", ep["target"]-ep["xy"]), ("radius", ep["radius"])):
                    sums[key] += float(np.square(values[ok]).sum(dtype=np.float64))
                error = np.sqrt(np.square((ep["xy"]-ep["target"])/ep["radius"]).sum(axis=1))
                ex = np.r_[0., np.cumsum(np.maximum(error-1., 0.), dtype=np.float64)]
                ot = np.r_[0, np.cumsum(error>1., dtype=np.int64)]
                e = (ex[candidates+FUTURE_MS]-ex[candidates])/FUTURE_MS
                o = (ot[candidates+FUTURE_MS]-ot[candidates])/FUTURE_MS
                q = e + o
            else:
                # No validation quality cutoff or quality ranking is calculated.
                e = o = q = np.full(len(candidates), np.nan, dtype=np.float64)
            episodes.extend([i]*len(candidates)); cuts.extend(candidates)
            scores.extend(q); outside.extend(o); excess.extend(e)
        if not train_ticks:
            raise ValueError("No valid TRAIN ticks for normalization")
        w = dict(episode_index=np.asarray(episodes,np.int64), cut_ms=np.asarray(cuts,np.int64),
                 quality_score=np.asarray(scores,np.float64), outside_fraction=np.asarray(outside,np.float64),
                 mean_positive_exceedance=np.asarray(excess,np.float64))
        w["keep95"] = np.zeros(len(cuts),bool); w["keep80"] = np.zeros(len(cuts),bool)
        groups = defaultdict(list)
        for j, e in enumerate(w["episode_index"]):
            if self.rows[e]["split"] == "train":
                groups[(self.people[e],self.rows[e]["family"])].append(j)
        self.quality_groups = []
        for (person,family), indices in sorted(groups.items()):
            order = sorted(indices, key=lambda j:(w["quality_score"][j], hashlib.sha256(
                f"tracking-control-quality:{TIE_SEED}:{self.rows[w['episode_index'][j]]['id']}:{w['cut_ms'][j]}".encode()).digest()))
            summary = dict(person=person,family=family,quality=distribution(w["quality_score"][indices]))
            for fraction, key in ((.95,"keep95"),(.8,"keep80")):
                count = math.ceil(fraction*len(order))
                w[key][order[:count]] = True
                summary[key] = dict(count=count,actual_fraction=count/len(order), cutoff=float(w["quality_score"][order[count-1]]),
                                    cutoff_ties_total=int(np.sum(w["quality_score"][indices] == w["quality_score"][order[count-1]])))
            self.quality_groups.append(summary)
        self.scales = dict(schema="abcurves.tracking_control.train_numerical_scales.v1", train_ticks=train_ticks,
                           units="source1-reference common angular counts; sample interval 1 ms", pooled_weighting="one weight per valid TRAIN tick; all TRAIN, before quality retention",
                           definitions={"velocity":"vector RMS of 1 ms delta", "position":"vector RMS of xy minus episode initial", "target_error":"vector RMS of target minus xy", "radius":"vector RMS of target semiaxes"},
                           **{key:max(math.sqrt(value/train_ticks),1e-6) for key,value in sums.items()})
        for a in w.values():
            a.setflags(write=False)
        self.windows, self.skipped_invalid = w, dict(skipped)
        return self


    def sampling_weights(self, indices):
        indices = np.asarray(indices,np.int64)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Sampling roster must not duplicate window indices")
        keys = []
        for j in indices:
            e = self.windows["episode_index"][j]; r = self.rows[e]
            keys.append((self.people[e],r["family"],r["guide"],r["parent"]))
        children = defaultdict(set); anchors = Counter(keys)
        for key in keys:
            for depth in range(4): children[key[:depth]].add(key[depth])
        weights = np.array([1./(math.prod(len(children[key[:depth]]) for depth in range(4))*anchors[key]) for key in keys],np.float64)
        if len(weights):
            weights /= weights.sum()
        return weights


def phase_of_cut(row,cut_ms):
    if "active_start_qpc" in row:
        if "replacement_for" not in row or row["split"]!="train":
            raise ValueError("Acquisition phase requires an explicit TRAIN replacement")
        if int(row["start_qpc"])+int(cut_ms)*10000<int(row["active_start_qpc"]):
            return "acquisition"
    return "active"

def select_phase_preserving(data,keep=.95,phase_balance=False):
    """Return global window indices, probabilities, JSON-serializable receipt.

    Must be called on the merged index to expose authentic acquisition. It
    never mutates data.windows or reads/ranks validation quality.
    """
    if not 0<keep<=1 or not isinstance(phase_balance,bool):
        raise ValueError("Need0<keep<=1 and Boolean phase_balance")
    data.build()
    w=data.windows
    candidates=[]
    phase={}
    groups=defaultdict(list)
    acquisition=[]
    for j,e in enumerate(w["episode_index"]):
        row=data.rows[int(e)]
        if row["split"]!="train":
            continue
        candidates.append(j)
        phase[j]=phase_of_cut(row,w["cut_ms"][j])
        if phase[j]=="acquisition":
            acquisition.append(j)
        else:
            groups[(data.people[int(e)],row["family"])].append(j)
    keep_indices=set(acquisition)
    group_receipts=[]
    for (person,family),ids in sorted(groups.items()):
        def sort_key(j):
            row=data.rows[int(w["episode_index"][j])]
            tie=hashlib.sha256(f"tracking-control-quality:{TIE_SEED}:{row['id']}:{w['cut_ms'][j]}".encode()).digest()
            return float(w["quality_score"][j]),tie
        ordered=sorted(ids,key=sort_key)
        count=math.ceil(keep*len(ordered))
        selected=ordered[:count]
        keep_indices.update(selected)
        group_receipts.append({"person":person,"family":family,"phase":"active","eligible":len(ids),
            "retained":count,"fraction_retained":count/len(ids),"cutoff":float(w["quality_score"][selected[-1]]),
            "quality_before":distribution(w["quality_score"][ids]),"quality_after":distribution(w["quality_score"][selected])})
    indices=np.asarray(sorted(keep_indices),np.int64)
    if not phase_balance:
        weights=data.sampling_weights(indices)
    else:
        keys=[]
        for j in indices:
            e=int(w["episode_index"][j]);row=data.rows[e]
            keys.append((data.people[e],phase[int(j)],row["family"],row["guide"],row["parent"]))
        children=defaultdict(set)
        anchors=Counter(keys)
        for key in keys:
            for depth in range(len(key)):
                children[key[:depth]].add(key[depth])
        weights=np.array([1/(math.prod(len(children[key[:d]]) for d in range(len(key)))*anchors[key]) for key in keys],np.float64)
        weights/=weights.sum()
    if len(np.unique(indices))!=len(indices) or not np.isclose(weights.sum(),1) or np.any(weights<=0):
        raise ValueError("Bad enrollment/sampling law")
    if not set(acquisition)<=set(map(int,indices)):
        raise ValueError("Acquisition loss")
    probability=defaultdict(float)
    for j,prob in zip(indices,weights):
        row=data.rows[int(w["episode_index"][j])]
        probability[(row["person"],phase[int(j)])]+=float(prob)
    enrollment=[]
    for person in sorted(set(data.people)):
        for ph in ("acquisition","active"):
            eligible=[j for j in candidates if data.rows[int(w["episode_index"][j])]["person"]==person and phase[j]==ph]
            selected=[j for j in eligible if j in keep_indices]
            enrollment.append({"person":person,"phase":ph,"eligible":len(eligible),"retained":len(selected),
                "excluded":len(eligible)-len(selected),"probability":probability[(person,ph)],
                "quality_before":distribution(w["quality_score"][eligible]),
                "quality_after":distribution(w["quality_score"][selected])})
    receipt={"schema":"abcurves.human_movement.phase_preserving_selection.v1","split":"train",
        "index_path":str(data.index_path),"index_sha256":data.index_sha256,
        "rule":"All eligible acquisition-origin cuts retained; active-origin cuts ranked afresh within each person/family by original1024ms quality score, ceil(keep*N), original deterministic tie seed.",
        "cut_phase":"Acquisition iff original cut QPC < authentic active_start_qpc; a future crossing into active remains an acquisition-origin window. Phase is selection metadata only.",
        "keep_active":keep,"phase_balance":phase_balance,
        "sampling":"equal person -> phase -> family -> guide -> parent -> anchor" if phase_balance else "unchanged equal person -> family -> guide -> parent -> anchor",
        "phase_model_input":False,"validation":"Not enrolled, quality not inspected or ranked.",
        "windows":len(indices),"acquisition_windows":len(acquisition),"active_groups":group_receipts,"enrollment":enrollment,
        "indices_sha256":hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "weights_sha256":hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest(),"script_sha256":sha(Path(__file__))}
    return indices,weights,receipt
