# esnf_mat_analyzer/core/container.py
from dataclasses import dataclass
from typing import Dict, Type, Any, Optional
import inspect

class RegistrationError(Exception):
    pass

class ResolutionError(Exception):
    pass

@dataclass
class ServiceDescriptor:
    """Describes a service registration in the container."""
    service_type: Type
    implementation: Type
    singleton: bool = False
    factory: Optional[callable] = None

class DependencyContainer:
    """Lightweight dependency injection container following SOLID principles.

    Features:
    - Interface-based registration
    - Automatic constructor injection
    - Singleton lifecycle management
    - Factory pattern support
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}

    def register(self, interface: Type, implementation: Type,
                singleton: bool = False, factory: Optional[callable] = None) -> None:
        """Register a service implementation for an interface.

        Args:
            interface: The interface type to register
            implementation: The concrete implementation type
            singleton: Whether to use singleton lifecycle
            factory: Optional factory function for complex instantiation

        Raises:
            RegistrationError: When interface/implementation mismatch detected
        """
        # Validate implementation conforms to interface
        if not issubclass(implementation, interface):
            if not hasattr(interface, '__protocols__'):  # Not a Protocol
                raise RegistrationError(f"{implementation} does not implement {interface}")

        self._services[interface] = ServiceDescriptor(
            interface, implementation, singleton, factory
        )

    def resolve(self, service_type: Type) -> Any:
        """Resolve a service instance with automatic dependency injection.

        Args:
            service_type: The interface type to resolve

        Returns:
            Configured service instance

        Raises:
            ResolutionError: When service cannot be resolved
        """
        if service_type in self._instances:
            return self._instances[service_type]

        if service_type not in self._services:
            raise ResolutionError(f"Service {service_type} not registered")

        descriptor = self._services[service_type]

        if descriptor.factory:
            instance = descriptor.factory(self)
        else:
            instance = self._create_instance(descriptor.implementation)

        if descriptor.singleton:
            self._instances[service_type] = instance

        return instance

    def _create_instance(self, implementation: Type) -> Any:
        """Create instance with automatic dependency injection."""
        signature = inspect.signature(implementation.__init__)
        dependencies = {}

        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue

            if param.annotation != inspect.Parameter.empty:
                dependencies[param_name] = self.resolve(param.annotation)

        return implementation(**dependencies)
