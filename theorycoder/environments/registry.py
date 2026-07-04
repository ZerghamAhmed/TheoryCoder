"""Game environment registry and factory."""
from typing import Any, Callable, Dict, Type

from theorycoder.environments.base import BaseGameEnvironment


class GameRegistry:
    """Registry for game environment classes.

    This allows dynamic registration and instantiation of game environments
    without hardcoding imports in the main codebase.
    """

    _registry: Dict[str, Callable[..., BaseGameEnvironment]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseGameEnvironment]) -> None:
        """Register a game environment factory.

        Args:
            name: Name of the game (e.g., 'babyai', 'sokoban')
            factory: Callable that creates the environment
        """
        cls._registry[name] = factory

    @classmethod
    def get(cls, name: str) -> Callable[..., BaseGameEnvironment]:
        """Get a game environment factory by name.

        Args:
            name: Name of the game

        Returns:
            Factory callable

        Raises:
            KeyError: If game not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Unknown game: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def create(
        cls,
        name: str,
        level_set: str,
        level_id: int,
        **kwargs
    ) -> BaseGameEnvironment:
        """Create a game environment instance.

        Args:
            name: Name of the game
            level_set: Level set identifier
            level_id: Level ID
            **kwargs: Additional arguments for the environment

        Returns:
            Environment instance
        """
        factory = cls.get(name)
        return factory(level_set=level_set, level_id=level_id, **kwargs)

    @classmethod
    def list_games(cls) -> list:
        """List all registered games."""
        return list(cls._registry.keys())


def register_game(name: str):
    """Decorator to register a game environment class.

    Usage:
        @register_game("my_game")
        class MyGameEnv(BaseGameEnvironment):
            ...

    Args:
        name: Name to register the game under
    """
    def decorator(cls):
        GameRegistry.register(name, cls)
        return cls
    return decorator


def get_game(name: str, level_set: str, level_id: int, **kwargs) -> BaseGameEnvironment:
    """Convenience function to create a game environment.

    Args:
        name: Name of the game
        level_set: Level set identifier
        level_id: Level ID
        **kwargs: Additional arguments

    Returns:
        Environment instance
    """
    return GameRegistry.create(name, level_set, level_id, **kwargs)


def register_default_games():
    """Register all default game environments.

    This function registers the standard games shipped with TheoryCoder.
    Call this at startup to make games available through the registry.
    """
    # Define lazy factories to avoid import errors if games aren't installed
    def _make_babyai(level_set, level_id, **kwargs):
        from babyai_env import BabyAI
        return BabyAI(level_set=level_set, level_id=level_id, **kwargs)

    def _make_doggo(level_set, level_id, **kwargs):
        from doggo_env import DoggoEnv
        return DoggoEnv(level_set=level_set, level_id=level_id, **kwargs)

    def _make_drunkdwarf(level_set, level_id, **kwargs):
        from drunkdwarf_env import DrunkDwarfEnv
        return DrunkDwarfEnv(level_set=level_set, level_id=level_id, **kwargs)

    def _make_boulderdash2(level_set, level_id, **kwargs):
        from boulderdash2_env import Boulderdash2Env
        return Boulderdash2Env(level_set=level_set, level_id=level_id, **kwargs)

    def _make_pb1(level_set, level_id, **kwargs):
        from pb1_env import pb1env
        return pb1env(level_set=level_set, level_id=level_id, **kwargs)

    def _make_sokoban(level_set, level_id, **kwargs):
        from sokoban_env import SokobanEnv
        return SokobanEnv(level_set=level_set, level_id=level_id, **kwargs)

    def _make_sokoban_full(level_set, level_id, **kwargs):
        from sokobanFULL_env import SokobanEnvFULL
        return SokobanEnvFULL(level_set=level_set, level_id=level_id, **kwargs)

    def _make_labyrinth(level_set, level_id, **kwargs):
        from labyrinth_env import LabyrinthEnv
        return LabyrinthEnv(level_set=level_set, level_id=level_id, **kwargs)

    def _make_cheesemaze(level_set, level_id, **kwargs):
        from cheesemaze_env import CheesemazeEnv
        return CheesemazeEnv(level_set=level_set, level_id=level_id, **kwargs)

    def _make_baba(level_set, level_id, **kwargs):
        from games import BabaIsYou
        return BabaIsYou(level_set=level_set, level_id=level_id, **kwargs)

    # Register all games
    GameRegistry.register("babyai", _make_babyai)
    GameRegistry.register("doggo", _make_doggo)
    GameRegistry.register("drunkdwarf", _make_drunkdwarf)
    GameRegistry.register("boulderdash2", _make_boulderdash2)
    GameRegistry.register("pb1", _make_pb1)
    GameRegistry.register("sokoban", _make_sokoban)
    GameRegistry.register("sokobanFULL", _make_sokoban_full)
    GameRegistry.register("labyrinth", _make_labyrinth)
    GameRegistry.register("cheesemaze", _make_cheesemaze)
    GameRegistry.register("baba", _make_baba)
