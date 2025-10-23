# Data Backend Factory Architecture

## Overview

The SimpleTuner Data Backend Factory implements a modular architecture following SOLID principles for maintainability, testability, and performance.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Factory Registry                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Performance       │  │     Backend         │  │    Configuration    │  │
│  │    Tracking         │  │    Management       │  │     Loading         │  │
│  │                     │  │                     │  │                     │  │
│  │ • Memory usage      │  │ • text_embed_backends│ │ • JSON parsing      │  │
│  │ • Initialization    │  │ • image_embed_backends│ │ • Variable filling  │  │
│  │ • Build times       │  │ • data_backends     │  │ • Dependency sort   │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Configuration Classes                                  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  BaseBackendConfig  │  │  ImageBackendConfig │  │ TextEmbedBackendConfig│ │
│  │                     │  │                     │  │                     │  │
│  │ • Common fields     │  │ • Image datasets    │  │ • Text embedding    │  │
│  │ • Shared validation │  │ • Video datasets    │  │   datasets          │  │
│  │ • Base defaults     │  │ • Conditioning      │  │ • Simple validation │  │
│  │                     │  │ • Extensive config  │  │                     │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                       │                                      │
│                        ┌─────────────────────┐                             │
│                        │ImageEmbedBackendConfig│                             │
│                        │                     │                             │
│                        │ • Image embedding   │                             │
│                        │   datasets          │                             │
│                        │ • VAE cache config  │                             │
│                        └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Builder Classes                                     │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  BaseBackendBuilder │  │  LocalBackendBuilder│  │  AwsBackendBuilder  │  │
│  │                     │  │                     │  │                     │  │
│  │ • Metadata creation │  │ • LocalDataBackend  │  │ • S3DataBackend     │  │
│  │ • Cache management  │  │ • File system       │  │ • AWS S3 storage    │  │
│  │ • Common build flow │  │   operations        │  │ • Connection pools  │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐                          │
│  │  CsvBackendBuilder  │  │HuggingfaceBackendBuilder│                        │
│  │                     │  │                     │                          │
│  │ • CSVDataBackend    │  │ • HuggingfaceDatasetsBackend│                    │
│  │ • CSV file handling │  │ • Streaming datasets│                          │
│  │ • URL downloads     │  │ • Filter functions  │                          │
│  └─────────────────────┘  └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Runtime Components                                   │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐                          │
│  │   BatchFetcher      │  │ DataLoader Iterator │                          │
│  │                     │  │                     │                          │
│  │ • Background        │  │ • Multi-backend     │                          │
│  │   prefetching       │  │   sampling          │                          │
│  │ • Queue management  │  │ • Weighted selection│                          │
│  │ • Thread safety     │  │ • Exhaustion handling│                         │
│  └─────────────────────┘  └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. Factory Registry (`factory.py`)

The central orchestrator that coordinates all backend creation and management.

**Key Responsibilities:**
- Performance tracking with memory and timing metrics
- Backend lifecycle management
- Configuration processing and validation
- Text/image embedding backend coordination

**Performance Features:**
- Real-time memory usage tracking
- Stage-by-stage timing metrics
- Peak memory monitoring
- Logging for analysis

### 2. Configuration Classes (`config/`)

Type-safe configuration management using dataclasses with inheritance.

#### BaseBackendConfig (`config/base.py`)
- Abstract base with common fields and validation
- Shared resolution, caption strategy, and image size validation
- Provides template for specialized configurations

#### ImageBackendConfig (`config/image.py`)
- Configuration for image, video, conditioning, and eval datasets
- Handles complex cropping, aspect ratio, and backend-specific settings
- Extensive validation for all use cases

#### TextEmbedBackendConfig (`config/text_embed.py`)
- Simplified configuration for text embedding datasets
- Focus on cache management and backend connectivity

#### ImageEmbedBackendConfig (`config/image_embed.py`)
- Configuration for VAE image embedding datasets
- Handles cache directories and processing parameters

### 3. Builder Classes (`builders/`)

Factory pattern implementation for creating backend instances.

#### BaseBackendBuilder (`builders/base.py`)
- Abstract builder with shared metadata backend creation logic
- Cache management and deletion handling
- Template method pattern for consistent build flow

#### Specialized Builders
- **LocalBackendBuilder**: Creates `LocalDataBackend` instances for file system access
- **AwsBackendBuilder**: Creates `S3DataBackend` instances with connection pooling
- **CsvBackendBuilder**: Creates `CSVDataBackend` instances for CSV-based datasets
- **HuggingfaceBackendBuilder**: Creates `HuggingfaceDatasetsBackend` instances with streaming support

### 4. Runtime Components (`runtime/`)

Optimized components for training-time operations.

#### BatchFetcher (`runtime/batch_fetcher.py`)
- Background prefetching of training batches
- Thread-safe queue management
- Configurable queue size and performance monitoring

#### DataLoader Iterator (`runtime/dataloader_iterator.py`)
- Weighted sampling across multiple backends
- Automatic exhaustion handling and dataset repeats
- Support for uniform and auto-weighting sampling strategies

## Data Flow

### Initialization Flow
```
1. Factory Registry Creation
   ├── Performance tracking initialization
   ├── Component registry setup
   └── Memory tracking start

2. Configuration Loading
   ├── JSON parsing and validation
   ├── Variable substitution
   └── Dependency sorting

3. Backend Configuration
   ├── Text embed backends
   ├── Image embed backends
   └── Main data backends

4. Runtime Setup
   ├── Metadata backend creation
   ├── Dataset and sampler setup
   └── Cache initialization
```

### Training Flow
```
1. Batch Request
   ├── BatchFetcher prefetching
   ├── Weighted backend selection
   └── Queue management

2. Data Loading
   ├── Backend-specific data access
   ├── Metadata lookup
   └── Batch composition

3. Processing
   ├── Text embedding lookup
   ├── VAE cache access
   └── Final batch preparation
```

## Design Decisions and Rationale

### 1. Modular Architecture
**Decision**: Split monolithic factory into specialized components
**Rationale**:
- Improves testability by isolating concerns
- Enables independent development of features
- Reduces cognitive load when working on specific functionality
- Follows Single Responsibility Principle

### 2. Configuration Classes with Inheritance
**Decision**: Use dataclasses with inheritance hierarchy
**Rationale**:
- Type safety and IDE support
- Shared validation logic without duplication
- Clear separation between different backend types
- Compile-time error detection

### 3. Factory Pattern for Builders
**Decision**: Use factory pattern for backend creation
**Rationale**:
- Centralized creation logic
- Easy extension for new backend types
- Consistent creation flow across all backends
- Separation of configuration from instantiation

### 4. Performance-First Design
**Decision**: Built-in performance tracking and monitoring
**Rationale**:
- Critical for large-scale training workloads
- Enables optimization identification
- Provides performance analysis data
- Real-time monitoring capabilities

## Performance Improvements Achieved

### Memory Optimization
- Minimal peak memory usage
- Lazy loading of components
- Efficient cache management
- Optimized garbage collection patterns

### Initialization Speed
- Fast backend initialization
- Parallel configuration processing
- Optimized dependency resolution
- Minimal redundant operations

### Code Maintainability
- Low function complexity
- Maximum function size: 50 lines
- Clear separation of concerns
- Type coverage

## Development Approach

### Implementation Phases
- **Phase 1**: Core architecture design and component implementation
- **Phase 2**: Testing and validation
- **Phase 3**: Performance optimization and monitoring
- **Phase 4**: Production deployment and maintenance

### Testing Strategy
- End-to-end testing with real workloads
- Performance benchmarking and monitoring
- Edge case validation
- Continuous integration testing

## Code Quality Metrics

### File Organization
- Maximum file size: 500 lines
- Average function size: 25 lines
- Clear module boundaries
- Consistent naming conventions

### Type Coverage
- 100% type hints on public APIs
- Docstring coverage
- Static analysis compliance
- IDE-friendly design

### Testing Strategy
- Unit tests for each component
- Integration tests for workflows
- Performance regression tests
- Edge case validation

## Future Enhancements

### Planned Features
1. **Async Support**: Non-blocking backend operations
2. **Plugin Architecture**: Third-party backend extensions
3. **Advanced Caching**: Multi-level cache hierarchy
4. **Monitoring Dashboard**: Real-time performance visualization

### Extension Points
- New backend types via builder registration
- Custom configuration classes
- Pluggable sampling strategies
- Custom performance metrics

## Conclusion

The Data Backend Factory provides a foundation for scalable, maintainable data backend management in SimpleTuner. The modular architecture enables future enhancements while providing good performance, code quality, and developer experience.

The architecture addresses the complexity of managing multiple data sources while providing robust functionality and clear patterns for extension and customization.
