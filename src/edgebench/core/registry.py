"""
FILENAME:       src/edgebench/core/registry.py

DESCRIPTION:    Model registry resolving an architecture name to an untrained nn.Module, across
                timm, torchvision and externally sourced implementations. Resolution records the
                provider and its version so that every checkpoint, export and benchmark result
                can be traced back to the exact implementation that produced it.

AUTHOR:         Kananelo Chabeli

DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from types import MappingProxyType
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from torch import nn

# Provider identifiers. Stored verbatim in checkpoint metadata, so renaming one invalidates the
# provenance of every artefact already produced.
SOURCE_EXTERNAL = "external"
SOURCE_TIMM = "timm"
SOURCE_TORCHVISION = "torchvision"

# Externally sourced implementations are consulted first. A project-local implementation is
# normally added because the library version is absent or known to differ from the paper, so it
# must take precedence; shadowing is made visible through ResolvedModel.also_available_in rather
# than being prevented.
DEFAULT_RESOLUTION_ORDER: tuple[str, ...] = (SOURCE_EXTERNAL, SOURCE_TIMM, SOURCE_TORCHVISION)


class RegistryError(RuntimeError):
    """Raised when an architecture cannot be resolved or a provider is misconfigured."""


@dataclass(frozen=True)
class ModelSpec:
    """
    Declarative request for one architecture.

    Deliberately excludes training and execution concerns such as batch size because this record
    describes WHICH network to build, not how to run it.

    Attributes:
        name: Architecture identifier as the provider knows it, for example "mobilenetv2_100".
        source: Provider to use. None resolves through DEFAULT_RESOLUTION_ORDER.
        kwargs: Constructor arguments forwarded to the provider, for example
            {"aux_logits": False}.
    """

    name: str
    source: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    # [AUDIT G2] frozen=True promised immutability, but kwargs stayed a mutable dict and the
    # generated __hash__ raised TypeError on it. kwargs is now a read-only view and hashing
    # uses a canonical rendering of it.
    # Before: (no __post_init__ / __hash__; kwargs stored as the dict passed in)
    def __post_init__(self) -> None:
        """Freeze kwargs into a read-only mapping.

        Returns:
            None.
        """
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))

    def __hash__(self) -> int:
        """Hash on name, source and a canonical rendering of kwargs.

        Returns:
            The hash value.
        """
        return hash((self.name, self.source, repr(sorted(self.kwargs.items()))))

    @classmethod
    def coerce(cls, value: "ModelSpec | str | Mapping[str, Any]") -> "ModelSpec":
        """Normalise the several shorthand forms used in configuration files.

        Accepting a bare string keeps a model list readable when most entries need no options,
        while a mapping allows per-model overrides without a separate schema.

        Args:
            value: A ModelSpec, an architecture name, or a mapping of ModelSpec fields.

        Returns:
            An equivalent ModelSpec.

        Raises:
            RegistryError: If the value is of an unsupported type or carries unknown fields.
        """
        if isinstance(value, ModelSpec):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, Mapping):
            unknown = set(value) - {"name", "source", "kwargs"}
            if unknown:
                raise RegistryError(f"unknown ModelSpec fields {sorted(unknown)}")
            if "name" not in value:
                raise RegistryError("ModelSpec mapping requires a 'name' field")
            return cls(name=value["name"], source=value.get("source"),
                       kwargs=dict(value.get("kwargs") or {}))
        raise RegistryError(f"cannot coerce {type(value).__name__} to ModelSpec")

    def as_dict(self) -> dict[str, Any]:
        """
        Serialise for storage in checkpoint metadata.

        Returns:
            Mapping of field name to value, with kwargs copied to a plain dict.
        """
        return {"name": self.name, "source": self.source, "kwargs": dict(self.kwargs)}


@dataclass
class ResolvedModel:
    """
    An instantiated architecture together with the provenance of its implementation.

    Attributes:
        module: The untrained network, with its classifier sized to num_classes.
        name: Architecture identifier used to build it.
        source: Provider that supplied the implementation.
        provider_version: Version string of that provider; a library release for timm and
            torchvision, a repository commit for external implementations.
        num_classes: Size of the classifier output.
        kwargs: Constructor arguments actually passed.
        also_available_in: Other providers that also carry this name. Non-empty indicates the
            chosen provider shadowed another, which affects which paper's variant was built.
    """

    module: nn.Module
    name: str
    source: str
    provider_version: str
    num_classes: int
    kwargs: dict[str, Any] = field(default_factory=dict)
    also_available_in: tuple[str, ...] = ()

    def provenance(self) -> dict[str, Any]:
        """Summarise the implementation origin for storage alongside weights.

        Returns:
            Mapping suitable for embedding in a checkpoint or results row. Excludes the module
            itself, which is not serialisable as metadata.
        """
        return {
            "name": self.name,
            "source": self.source,
            "provider_version": self.provider_version,
            "num_classes": self.num_classes,
            "kwargs": dict(self.kwargs),
            "also_available_in": list(self.also_available_in),
        }


class ModelProvider(ABC):
    """Interface implemented by each source of architecture definitions."""

    #: Identifier recorded in checkpoint metadata.
    source: str = ""

    @abstractmethod
    def available(self) -> bool:
        """Report whether the provider's backing library is importable.

        Returns:
            True when the provider can be queried. A provider whose library is absent is
            skipped during resolution rather than raising, so a machine that compiles but does
            not train need not install every framework.
        """

    @abstractmethod
    def version(self) -> str:
        """Report the provider's version for provenance records.

        Returns:
            A version string, or "unavailable" when the backing library cannot be imported.
        """

    @abstractmethod
    def has(self, name: str) -> bool:
        """Report whether this provider defines the given architecture.

        Args:
            name: Architecture identifier.

        Returns:
            True if the provider can build it.
        """

    @abstractmethod
    def build(self, name: str, num_classes: int, **kwargs: Any) -> nn.Module:
        """Instantiate the architecture with randomly initialised weights.

        Args:
            name: Architecture identifier.
            num_classes: Size of the classifier output.
            **kwargs: Provider-specific constructor arguments.

        Returns:
            The untrained module.

        Raises:
            RegistryError: If instantiation fails.
        """

    @abstractmethod
    def list_models(self, pattern: str | None = None) -> list[str]:
        """List architecture names this provider offers.

        Args:
            pattern: Optional glob filter, for example "*ghost*".

        Returns:
            Sorted architecture names, empty when the provider is unavailable.
        """

    def effective_kwargs(self, name: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve the constructor arguments actually used for an architecture.

        Providers override this to force arguments the trainer depends on. The result is what
        gets recorded in provenance, so a forced argument is never invisible.

        Args:
            name: Architecture identifier.
            kwargs: Arguments requested by the spec.

        Returns:
            The arguments to build with.

        Raises:
            RegistryError: If a requested argument conflicts with a forced one.
        """
        return dict(kwargs)


def _try_import(module_name: str):
    """Import a module, returning None when it is absent or broken.

    Args:
        module_name: Fully qualified module name.

    Returns:
        The imported module, or None if it cannot be imported.
    """
    # [AUDIT G3] A broken install (e.g. a timm built against another torch) raises errors
    # other than ImportError; those used to crash every resolution. They now disable the one
    # provider, with the reason printed rather than hidden.
    # Before:
    #   try:
    #       return importlib.import_module(module_name)
    #   except ImportError:
    #       return None
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
    except Exception as exc:
        print(f"  WARNING: {module_name} is installed but failed to import "
              f"({type(exc).__name__}: {exc}); treating it as unavailable")
        return None


class TimmProvider(ModelProvider):
    """Architectures supplied by the timm library."""

    source = SOURCE_TIMM

    def __init__(self) -> None:
        self._timm = None
        self._probed = False

    def _lib(self):
        """Import timm on first use and cache the result.

        Deferred so that importing the registry on a device that only runs compiled binaries
        does not pull in a large and unnecessary dependency.

        Returns:
            The timm module, or None when it is not installed.
        """
        if not self._probed:
            self._timm = _try_import("timm")
            self._probed = True
        return self._timm

    def available(self) -> bool:
        """Report whether timm is importable.

        Returns:
            True when timm is installed.
        """
        return self._lib() is not None

    def version(self) -> str:
        """Report the installed timm version.

        Returns:
            The version string, or "unavailable".
        """
        lib = self._lib()
        return getattr(lib, "__version__", "unknown") if lib else "unavailable"

    def has(self, name: str) -> bool:
        """Report whether timm defines the architecture.

        Args:
            name: Architecture identifier.

        Returns:
            True if timm knows the name.
        """
        lib = self._lib()
        return bool(lib and lib.is_model(name))

    def build(self, name: str, num_classes: int, **kwargs: Any) -> nn.Module:
        """Instantiate a timm architecture without pretrained weights.

        Args:
            name: Architecture identifier.
            num_classes: Size of the classifier output.
            **kwargs: Forwarded to timm.create_model.

        Returns:
            The untrained module.

        Raises:
            RegistryError: If timm is unavailable or construction fails.
        """
        lib = self._lib()
        if lib is None:
            raise RegistryError("timm is not installed")
        try:
            # pretrained=False is non-negotiable for this framework: the comparison is between
            # architectures trained under one recipe, and pretrained weights would reintroduce
            # each model's original and differing training procedure.
            return lib.create_model(name, pretrained=False, num_classes=num_classes, **kwargs)
        except Exception as exc:
            raise RegistryError(f"timm failed to build {name!r}: {exc}") from exc

    def list_models(self, pattern: str | None = None) -> list[str]:
        """List timm architecture names.

        Args:
            pattern: Optional glob filter.

        Returns:
            Sorted architecture names.
        """
        lib = self._lib()
        if lib is None:
            return []
        return sorted(lib.list_models(pattern) if pattern else lib.list_models())


class TorchvisionProvider(ModelProvider):
    """
    Classification architectures supplied by torchvision.
    """

    source = SOURCE_TORCHVISION

    # Architectures whose auxiliary classifier heads make the training forward pass return a
    # tuple. The shared trainer optimises a single head, so these are built single-headed and
    # the forced argument is recorded in provenance.
    SINGLE_HEAD_KWARGS: Mapping[str, Mapping[str, Any]] = {
        "inception_v3": {"aux_logits": False},
        "googlenet": {"aux_logits": False},
    }

    def __init__(self) -> None:
        self._tv = None
        self._probed = False
        self._names: frozenset[str] | None = None

    def _lib(self):
        """Import torchvision on first use and cache the result.

        Returns:
            The torchvision module, or None when it is not installed.
        """
        if not self._probed:
            self._tv = _try_import("torchvision")
            self._probed = True
        return self._tv

    def _model_names(self) -> frozenset[str]:
        """List torchvision's classification models, cached.

        Restricted to torchvision.models so that a detection or segmentation entry point cannot
        be matched by a name collision.

        Returns:
            The set of classification architecture names, empty when torchvision is absent.
        """
        # [AUDIT G3] list_models(module=...) only exists from torchvision 0.14; on older
        # releases this raised AttributeError inside every resolve(). Fall back to the public
        # lowercase builders in torchvision.models.
        # Before:
        #   self._names = (frozenset(lib.models.list_models(module=lib.models))
        #                  if lib else frozenset())
        if self._names is None:
            lib = self._lib()
            if lib is None:
                self._names = frozenset()
            elif hasattr(lib.models, "list_models"):
                self._names = frozenset(lib.models.list_models(module=lib.models))
            else:
                self._names = frozenset(
                    n for n, obj in vars(lib.models).items()
                    if callable(obj) and n.islower() and not n.startswith("_"))
        return self._names

    def available(self) -> bool:
        """Report whether torchvision is importable.

        Returns:
            True when torchvision is installed.
        """
        return self._lib() is not None

    def version(self) -> str:
        """Report the installed torchvision version.

        Returns:
            The version string, or "unavailable".
        """
        lib = self._lib()
        return getattr(lib, "__version__", "unknown") if lib else "unavailable"

    def has(self, name: str) -> bool:
        """Report whether torchvision defines the classification architecture.

        Args:
            name: Architecture identifier.

        Returns:
            True if torchvision knows the name.
        """
        return name in self._model_names()

    def effective_kwargs(self, name: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Force single-head construction for architectures with auxiliary classifiers.

        Args:
            name: Architecture identifier.
            kwargs: Arguments requested by the spec.

        Returns:
            The requested arguments plus any forced ones.

        Raises:
            RegistryError: If the spec explicitly asks for an auxiliary head.
        """
        # [AUDIT T9] In train mode torchvision's inception_v3 and googlenet returned a
        # namedtuple and crashed the trainer at logits.float(). Unwrapping the tuple in the
        # trainer would silently drop a head the model was built with, so the head is
        # disabled here, recorded in provenance, and an explicit request for it fails loudly.
        # Before: (kwargs were passed through unchanged)
        out = dict(kwargs)
        for key, forced in self.SINGLE_HEAD_KWARGS.get(name, {}).items():
            if key in out and out[key] != forced:
                raise RegistryError(
                    f"{name!r} with {key}={out[key]!r} is not supported: the trainer optimises "
                    f"a single classifier head, so {key} must be {forced!r}")
            out[key] = forced
        return out

    def build(self, name: str, num_classes: int, **kwargs: Any) -> nn.Module:
        """Instantiate a torchvision architecture without pretrained weights.

        Args:
            name: Architecture identifier.
            num_classes: Size of the classifier output.
            **kwargs: Forwarded to torchvision.models.get_model.

        Returns:
            The untrained module.

        Raises:
            RegistryError: If torchvision is unavailable or construction fails.
        """
        lib = self._lib()
        if lib is None:
            raise RegistryError("torchvision is not installed")
        try:
            return lib.models.get_model(name, weights=None, num_classes=num_classes, **kwargs)
        except Exception as exc:
            raise RegistryError(f"torchvision failed to build {name!r}: {exc}") from exc

    def list_models(self, pattern: str | None = None) -> list[str]:
        """List torchvision classification architecture names.

        Args:
            pattern: Optional glob filter.

        Returns:
            Sorted architecture names.
        """
        import fnmatch
        names = sorted(self._model_names())
        return fnmatch.filter(names, pattern) if pattern else names


@dataclass(frozen=True)
class ExternalEntry:
    """A manually registered architecture implementation.

    Attributes:
        name: Architecture identifier.
        builder: Callable returning an nn.Module, invoked as builder(num_classes, **kwargs).
        repo: Source repository, recorded for provenance.
        commit: Commit or tag pinning the implementation. Required: an unpinned external
            implementation makes a result unreproducible, since the upstream code may change.
        paper: Optional citation for the architecture.
        notes: Optional free text, for instance a deviation from the published description.
    """

    name: str
    builder: Callable[..., nn.Module]
    repo: str
    commit: str
    paper: str | None = None
    notes: str | None = None


class ExternalProvider(ModelProvider):
    """Architectures registered in-process from vendored or third-party repositories.

    Exists because several lineages in this study have no timm or torchvision implementation.
    Builders are stored as callables and invoked lazily, so registering an architecture does not
    import its repository until that architecture is actually requested.
    """

    source = SOURCE_EXTERNAL

    def __init__(self) -> None:
        self._entries: dict[str, ExternalEntry] = {}

    def register(self, entry: ExternalEntry, *, overwrite: bool = False) -> None:
        """Add an implementation to the provider.

        Args:
            entry: The implementation record.
            overwrite: Permit replacing an existing registration under the same name.

        Returns:
            None.

        Raises:
            RegistryError: If the name is already registered and overwrite is False.
        """
        if entry.name in self._entries and not overwrite:
            raise RegistryError(
                f"{entry.name!r} is already registered externally; pass overwrite=True to replace"
            )
        self._entries[entry.name] = entry

    def entry(self, name: str) -> ExternalEntry:
        """Retrieve the registration record for an architecture.

        Args:
            name: Architecture identifier.

        Returns:
            The registration record.

        Raises:
            RegistryError: If the name is not registered.
        """
        try:
            return self._entries[name]
        except KeyError as exc:
            raise RegistryError(f"{name!r} is not registered externally") from exc

    def available(self) -> bool:
        """Report whether any external implementation is registered.

        Returns:
            True when at least one registration exists.
        """
        return bool(self._entries)

    def version(self) -> str:
        """Report a version string for the provider as a whole.

        Returns:
            A fixed label; per-architecture commits are reported by build_version instead,
            because each registration pins its own repository.
        """
        return f"external:{len(self._entries)}-registered"

    def build_version(self, name: str) -> str:
        """Report the pinned commit for one registered architecture.

        Args:
            name: Architecture identifier.

        Returns:
            A "repo@commit" string identifying the exact implementation.

        Raises:
            RegistryError: If the name is not registered.
        """
        e = self.entry(name)
        return f"{e.repo}@{e.commit}"

    def has(self, name: str) -> bool:
        """Report whether the architecture is registered.

        Args:
            name: Architecture identifier.

        Returns:
            True if a registration exists.
        """
        return name in self._entries

    def build(self, name: str, num_classes: int, **kwargs: Any) -> nn.Module:
        """Invoke the registered builder.

        Args:
            name: Architecture identifier.
            num_classes: Size of the classifier output.
            **kwargs: Forwarded to the builder.

        Returns:
            The untrained module.

        Raises:
            RegistryError: If the name is unregistered, the builder fails, or the builder
                returns something other than an nn.Module.
        """
        entry = self.entry(name)
        try:
            module = entry.builder(num_classes=num_classes, **kwargs)
        except Exception as exc:
            raise RegistryError(f"external builder for {name!r} failed: {exc}") from exc
        if not isinstance(module, nn.Module):
            raise RegistryError(
                f"external builder for {name!r} returned {type(module).__name__}, expected nn.Module"
            )
        return module

    def list_models(self, pattern: str | None = None) -> list[str]:
        """List registered architecture names.

        Args:
            pattern: Optional glob filter.

        Returns:
            Sorted architecture names.
        """
        import fnmatch
        names = sorted(self._entries)
        return fnmatch.filter(names, pattern) if pattern else names

    def describe_all(self) -> list[dict[str, Any]]:
        """Summarise every registration.

        Returns:
            One mapping per registration, excluding the builder callable.
        """
        return [{"name": e.name, "repo": e.repo, "commit": e.commit,
                 "paper": e.paper, "notes": e.notes}
                for e in sorted(self._entries.values(), key=lambda x: x.name)]


class ModelRegistry:
    """
    Resolves architecture names across an ordered set of providers.

    Constructed explicitly so tests can supply their own providers; module-level helpers wrap a
    shared default instance for ordinary use.
    """

    def __init__(self, providers: Sequence[ModelProvider] | None = None,
                 order: Sequence[str] | None = None) -> None:
        """Initialise the registry.

        Args:
            providers: Providers to consult. Defaults to external, timm and torchvision.
            order: Provider identifiers in resolution priority order. Defaults to
                DEFAULT_RESOLUTION_ORDER.
        """
        # [AUDIT G1] With a custom provider list that lacked an ExternalProvider,
        # register_external wrote into an orphan provider that resolution never consulted.
        # One is now always part of the provider set.
        # Before:
        #   self.external = ExternalProvider()
        #   if providers is None:
        #       providers = [self.external, TimmProvider(), TorchvisionProvider()]
        #   else:
        #       for p in providers:
        #           if isinstance(p, ExternalProvider):
        #               self.external = p
        #   self._providers: dict[str, ModelProvider] = {p.source: p for p in providers}
        self.external = ExternalProvider()
        if providers is None:
            providers = [self.external, TimmProvider(), TorchvisionProvider()]
        else:
            providers = list(providers)
            supplied = [p for p in providers if isinstance(p, ExternalProvider)]
            if supplied:
                self.external = supplied[0]
            else:
                providers.append(self.external)
        self._providers: dict[str, ModelProvider] = {p.source: p for p in providers}
        self._order = tuple(order or DEFAULT_RESOLUTION_ORDER)

    # ---- registration -------------------------------------------------------------------

    def register_external(self, name: str, builder: Callable[..., nn.Module], *,
                          repo: str, commit: str, paper: str | None = None,
                          notes: str | None = None, overwrite: bool = False) -> None:
        """Register an architecture sourced from outside timm and torchvision.

        The builder is stored uncalled, so the repository it depends on is imported only when
        the architecture is actually requested. A missing repository therefore breaks exactly
        one model rather than the whole run.

        Args:
            name: Architecture identifier to register under.
            builder: Callable invoked as builder(num_classes=..., **kwargs).
            repo: Source repository identifier, for example "huawei-noah/Efficient-AI-Backbones".
            commit: Commit hash or tag pinning the implementation.
            paper: Optional citation.
            notes: Optional free text, such as a deviation from the published architecture.
            overwrite: Permit replacing an existing registration.

        Returns:
            None.

        Raises:
            RegistryError: If the name is already registered and overwrite is False.
        """
        self.external.register(
            ExternalEntry(name=name, builder=builder, repo=repo, commit=commit,
                          paper=paper, notes=notes),
            overwrite=overwrite,
        )

    # ---- queries ------------------------------------------------------------------------

    def providers_with(self, name: str) -> list[str]:
        """List every provider that defines an architecture.

        Args:
            name: Architecture identifier.

        Returns:
            Provider identifiers in resolution order. More than one entry means the name is
            ambiguous and the first will shadow the rest.
        """
        return [s for s in self._order
                if (p := self._providers.get(s)) and p.available() and p.has(name)]

    def is_available(self, name: str, source: str | None = None) -> bool:
        """Report whether an architecture can be built.

        Args:
            name: Architecture identifier.
            source: Restrict the check to one provider.

        Returns:
            True if some permitted provider defines the name.
        """
        if source:
            p = self._providers.get(source)
            return bool(p and p.available() and p.has(name))
        return bool(self.providers_with(name))

    def list_models(self, pattern: str | None = None,
                    source: str | None = None) -> dict[str, list[str]]:
        """List available architectures grouped by provider.

        Args:
            pattern: Optional glob filter, for example "*ghost*".
            source: Restrict to one provider.

        Returns:
            Mapping of provider identifier to sorted architecture names. Providers whose
            library is absent are omitted.
        """
        out: dict[str, list[str]] = {}
        for s in self._order:
            if source and s != source:
                continue
            p = self._providers.get(s)
            if p and p.available():
                names = p.list_models(pattern)
                if names:
                    out[s] = names
        return out

    def describe(self, name: str) -> dict[str, Any]:
        """Summarise how an architecture would be resolved, without building it.

        Args:
            name: Architecture identifier.

        Returns:
            Mapping describing the resolved provider, its version, and any shadowed providers.
            External entries additionally carry their repository, commit and citation.
        """
        found = self.providers_with(name)
        if not found:
            return {"name": name, "available": False, "candidates": []}
        chosen = found[0]
        info: dict[str, Any] = {
            "name": name,
            "available": True,
            "source": chosen,
            "provider_version": self._provider_version(chosen, name),
            "candidates": found,
            "also_available_in": found[1:],
        }
        if chosen == SOURCE_EXTERNAL:
            e = self.external.entry(name)
            info.update(repo=e.repo, commit=e.commit, paper=e.paper, notes=e.notes)
        return info

    # ---- resolution ---------------------------------------------------------------------

    def _provider_version(self, source: str, name: str) -> str:
        """Report the version string to record for one resolution.

        Args:
            source: Provider identifier.
            name: Architecture identifier.

        Returns:
            For external implementations, the per-architecture "repo@commit"; otherwise the
            provider's library version.
        """
        p = self._providers[source]
        if isinstance(p, ExternalProvider):
            return p.build_version(name)
        return p.version()

    def resolve(self, spec: "ModelSpec | str | Mapping[str, Any]",
                num_classes: int) -> ResolvedModel:
        """Build an architecture and record which implementation was used.

        Args:
            spec: A ModelSpec or any form accepted by ModelSpec.coerce.
            num_classes: Size of the classifier output.

        Returns:
            The instantiated module together with its provenance.

        Raises:
            RegistryError: If num_classes is not positive, the requested provider is
                unavailable, or no provider defines the architecture.
        """
        spec = ModelSpec.coerce(spec)
        if num_classes <= 0:
            raise RegistryError(f"num_classes must be positive, got {num_classes}")

        if spec.source:
            provider = self._providers.get(spec.source)
            if provider is None:
                raise RegistryError(
                    f"unknown source {spec.source!r}; known: {sorted(self._providers)}")
            if not provider.available():
                raise RegistryError(f"source {spec.source!r} is not installed")
            if not provider.has(spec.name):
                raise RegistryError(
                    f"{spec.name!r} not found in {spec.source!r}. "
                    f"Available elsewhere: {self.providers_with(spec.name) or 'nowhere'}")
            chosen = spec.source
            others: tuple[str, ...] = tuple(s for s in self.providers_with(spec.name)
                                            if s != chosen)
        else:
            found = self.providers_with(spec.name)
            if not found:
                raise RegistryError(
                    f"{spec.name!r} not found in any provider "
                    f"({', '.join(self._order)}). Register it with register_external() if it "
                    f"comes from an outside repository.")
            chosen, others = found[0], tuple(found[1:])

        # [AUDIT T9] Build with, and record, the provider's effective kwargs so a forced
        # argument such as aux_logits=False appears in every checkpoint's provenance.
        # Before:
        #   module = self._providers[chosen].build(spec.name, num_classes, **dict(spec.kwargs))
        #   ... kwargs=dict(spec.kwargs),
        provider = self._providers[chosen]
        kwargs = provider.effective_kwargs(spec.name, spec.kwargs)
        module = provider.build(spec.name, num_classes, **kwargs)
        return ResolvedModel(
            module=module,
            name=spec.name,
            source=chosen,
            provider_version=self._provider_version(chosen, spec.name),
            num_classes=num_classes,
            kwargs=kwargs,
            also_available_in=others,
        )

#TO DO: Implement BackboneProvider and DetectorProvider classes, 
# following the same pattern as TimmProvider and TorchvisionProvider, 
# to support additional model sources and types.


def add_repo_to_path(path: "str | Path") -> Path:
    """Prepend a vendored repository to sys.path so its modules become importable.

    Intended for use inside an external builder, keeping the path change local to the one
    architecture that needs it rather than applied at import time.

    Args:
        path: Directory containing the repository's importable packages.

    Returns:
        The resolved directory.

    Raises:
        RegistryError: If the directory does not exist.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise RegistryError(f"repository path does not exist: {p}")
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return p


# Shared instance for ordinary use. Tests should construct their own ModelRegistry.
_REGISTRY: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the process-wide registry, constructing it on first use.

    Returns:
        The shared ModelRegistry.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY


def set_registry(registry: ModelRegistry) -> None:
    """Replace the process-wide registry.

    Args:
        registry: The instance to install.

    Returns:
        None.
    """
    global _REGISTRY
    _REGISTRY = registry


def resolve_model(spec: "ModelSpec | str | Mapping[str, Any]",
                  num_classes: int) -> ResolvedModel:
    """Resolve an architecture through the shared registry.

    Args:
        spec: A ModelSpec or any form accepted by ModelSpec.coerce.
        num_classes: Size of the classifier output.

    Returns:
        The instantiated module together with its provenance.

    Raises:
        RegistryError: If the architecture cannot be resolved.
    """
    return get_registry().resolve(spec, num_classes)