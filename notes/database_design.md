# Empire.AI Database Design & Knowledge Base Architecture

*Design Date: 2025-11-14*  
*Status: Complete*  
*Technology Stack: SQLModel + PostgreSQL + Repository Pattern*

## Overview

This document outlines the comprehensive database design for Empire.AI's multi-agent music video creation system. The design implements a robust knowledge base that supports agent coordination, project management, and shared workspace functionality using SQLModel with PostgreSQL.

## Design Philosophy

### Core Principles
1. **Agent-Centric Design**: Database structure optimized for multi-agent workflows
2. **Type Safety**: Full SQLModel integration with Pydantic validation
3. **Scalable Architecture**: Repository pattern with Unit of Work for transaction management
4. **Knowledge Persistence**: Comprehensive storage for research, concepts, and production data
5. **Audit Trail**: Complete tracking of agent activities and decision processes

### Technology Stack
- **Database**: PostgreSQL 15+ with UUID extension
- **ORM**: SQLModel (Pydantic + SQLAlchemy)
- **Pattern**: Repository + Unit of Work
- **Connection**: AsyncPG for async operations
- **Migration**: Alembic for schema versioning

## Database Schema Design

### Schema Architecture

```mermaid
erDiagram
    %% Core Project Management
    PROJECTS {
        uuid id PK
        string name
        string description
        enum status
        jsonb user_preferences
        jsonb quality_requirements
        timestamp created_at
        timestamp updated_at
        uuid created_by_agent_id FK
    }
    
    %% Agent Management
    AGENTS {
        uuid id PK
        string name
        enum agent_type
        string version
        jsonb configuration
        jsonb capabilities
        enum status
        timestamp last_active
        jsonb performance_metrics
    }
    
    %% Music Research Data
    ARTIST_PROFILES {
        uuid id PK
        uuid project_id FK
        string name
        text[] genres
        jsonb audio_features
        text biography
        text[] similar_artists
        text cultural_context
        jsonb meta
        timestamp created_at
        uuid created_by_agent_id FK
    }
    
    SONG_ANALYSES {
        uuid id PK
        uuid project_id FK
        uuid artist_profile_id FK
        string title
        string artist_name
        text lyrics
        text[] themes
        string mood
        jsonb musical_characteristics
        jsonb audio_features
        jsonb annotations
        jsonb meta
        timestamp created_at
        uuid created_by_agent_id FK
    }
    
    %% Visual Concept Data
    VISUAL_CONCEPTS {
        uuid id PK
        uuid project_id FK
        string name
        text description
        string style
        string mood
        string color_palette
        text[] reference_images
        text[] generated_images
        float confidence_score
        jsonb style_parameters
        jsonb meta
        timestamp created_at
        uuid created_by_agent_id FK
    }
    
    REFERENCE_IMAGES {
        uuid id PK
        uuid visual_concept_id FK
        string url
        string source
        text description
        jsonb image_metadata
        enum image_type
        timestamp created_at
    }
    
    %% Video Production Data
    VIDEO_SEGMENTS {
        uuid id PK
        uuid project_id FK
        uuid visual_concept_id FK
        string name
        text description
        int duration_seconds
        string style
        string operation_id
        enum status
        string video_path
        jsonb generation_parameters
        jsonb quality_metrics
        jsonb meta
        timestamp created_at
        uuid created_by_agent_id FK
    }
    
    VIDEO_SEQUENCES {
        uuid id PK
        uuid project_id FK
        string name
        text description
        int total_duration
        jsonb segment_order
        string final_video_path
        enum status
        jsonb meta
        timestamp created_at
        timestamp completed_at
    }
    
    %% Agent Activity & Workflow
    AGENT_ACTIVITIES {
        uuid id PK
        uuid agent_id FK
        uuid project_id FK
        string operation
        text description
        enum status
        jsonb input_data
        jsonb output_data
        jsonb error_details
        float execution_time
        timestamp started_at
        timestamp completed_at
        string correlation_id
    }
    
    WORKFLOW_EXECUTIONS {
        uuid id PK
        uuid project_id FK
        enum workflow_pattern
        enum status
        jsonb step_results
        jsonb error_log
        float total_execution_time
        timestamp started_at
        timestamp completed_at
        string initiated_by
    }
    
    %% Quality Control
    QUALITY_ASSESSMENTS {
        uuid id PK
        uuid project_id FK
        string content_type
        uuid content_id
        jsonb quality_scores
        float overall_score
        boolean passed
        text[] issues
        text[] recommendations
        timestamp assessed_at
        uuid assessed_by_agent_id FK
    }
    
    %% Knowledge Base & Shared Workspace
    KNOWLEDGE_ENTRIES {
        uuid id PK
        uuid project_id FK
        enum entry_type
        string title
        text content
        jsonb structured_data
        text[] tags
        jsonb meta
        timestamp created_at
        timestamp updated_at
        uuid created_by_agent_id FK
        uuid updated_by_agent_id FK
    }
    
    SHARED_RESOURCES {
        uuid id PK
        uuid project_id FK
        string name
        enum resource_type
        string file_path
        string url
        jsonb metadata
        enum access_level
        timestamp created_at
        uuid created_by_agent_id FK
    }
    
    %% Relationships
    PROJECTS ||--o{ ARTIST_PROFILES : "has"
    PROJECTS ||--o{ SONG_ANALYSES : "contains"
    PROJECTS ||--o{ VISUAL_CONCEPTS : "includes"
    PROJECTS ||--o{ VIDEO_SEGMENTS : "produces"
    PROJECTS ||--o{ VIDEO_SEQUENCES : "creates"
    PROJECTS ||--o{ AGENT_ACTIVITIES : "logs"
    PROJECTS ||--o{ WORKFLOW_EXECUTIONS : "executes"
    PROJECTS ||--o{ QUALITY_ASSESSMENTS : "evaluates"
    PROJECTS ||--o{ KNOWLEDGE_ENTRIES : "accumulates"
    PROJECTS ||--o{ SHARED_RESOURCES : "stores"
    
    ARTIST_PROFILES ||--o{ SONG_ANALYSES : "analyzed_in"
    VISUAL_CONCEPTS ||--o{ REFERENCE_IMAGES : "references"
    VISUAL_CONCEPTS ||--o{ VIDEO_SEGMENTS : "inspires"
    
    AGENTS ||--o{ AGENT_ACTIVITIES : "performs"
    AGENTS ||--o{ PROJECTS : "creates"
    AGENTS ||--o{ ARTIST_PROFILES : "researches"
    AGENTS ||--o{ SONG_ANALYSES : "analyzes"
    AGENTS ||--o{ VISUAL_CONCEPTS : "develops"
    AGENTS ||--o{ VIDEO_SEGMENTS : "produces"
    AGENTS ||--o{ QUALITY_ASSESSMENTS : "assesses"
    AGENTS ||--o{ KNOWLEDGE_ENTRIES : "contributes"
    AGENTS ||--o{ SHARED_RESOURCES : "manages"
```

## SQLModel Implementation

### Base Models

```python
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func

# Enums
class ProjectStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"

class AgentType(str, Enum):
    MUSIC_RESEARCH = "music_research"
    VISUAL_CONCEPT = "visual_concept"
    VIDEO_PRODUCTION = "video_production"
    PROJECT_ORCHESTRATOR = "project_orchestrator"
    QUALITY_CONTROLLER = "quality_controller"

class AgentStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    OFFLINE = "offline"

class WorkflowPattern(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ITERATIVE = "iterative"
    CUSTOM = "custom"

class ActivityStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ContentType(str, Enum):
    ARTIST_PROFILE = "artist_profile"
    SONG_ANALYSIS = "song_analysis"
    VISUAL_CONCEPT = "visual_concept"
    VIDEO_SEGMENT = "video_segment"
    VIDEO_SEQUENCE = "video_sequence"

class EntryType(str, Enum):
    RESEARCH_NOTE = "research_note"
    CONCEPT_IDEA = "concept_idea"
    TECHNICAL_SPEC = "technical_spec"
    QUALITY_REPORT = "quality_report"
    DECISION_LOG = "decision_log"
    LESSON_LEARNED = "lesson_learned"

class ResourceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    CONFIG = "config"
    DATASET = "dataset"

class AccessLevel(str, Enum):
    PUBLIC = "public"
    PROJECT = "project"
    AGENT = "agent"
    RESTRICTED = "restricted"

# Base Model with Common Fields
class EmpireBaseModel(SQLModel):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_column_kwargs={"server_default": func.gen_random_uuid()}
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"server_default": func.now()}
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"onupdate": func.now()}
    )
    meta: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
```

### Core Entity Models

```python
# Agent Management
class Agent(EmpireBaseModel, table=True):
    __tablename__ = "agents"
    
    name: str = Field(index=True)
    agent_type: AgentType = Field(index=True)
    version: str
    configuration: Dict[str, Any] = Field(sa_column=Column(JSON))
    capabilities: Dict[str, Any] = Field(sa_column=Column(JSON))
    status: AgentStatus = Field(default=AgentStatus.OFFLINE, index=True)
    last_active: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    
    # Relationships
    projects: List["Project"] = Relationship(back_populates="created_by_agent")
    activities: List["AgentActivity"] = Relationship(back_populates="agent")
    artist_profiles: List["ArtistProfile"] = Relationship(back_populates="created_by_agent")
    song_analyses: List["SongAnalysis"] = Relationship(back_populates="created_by_agent")
    visual_concepts: List["VisualConcept"] = Relationship(back_populates="created_by_agent")
    video_segments: List["VideoSegment"] = Relationship(back_populates="created_by_agent")
    quality_assessments: List["QualityAssessment"] = Relationship(back_populates="assessed_by_agent")
    knowledge_entries_created: List["KnowledgeEntry"] = Relationship(
        back_populates="created_by_agent",
        sa_relationship_kwargs={"foreign_keys": "[KnowledgeEntry.created_by_agent_id]"}
    )
    knowledge_entries_updated: List["KnowledgeEntry"] = Relationship(
        back_populates="updated_by_agent",
        sa_relationship_kwargs={"foreign_keys": "[KnowledgeEntry.updated_by_agent_id]"}
    )
    shared_resources: List["SharedResource"] = Relationship(back_populates="created_by_agent")

# Project Management
class Project(EmpireBaseModel, table=True):
    __tablename__ = "projects"
    
    name: str = Field(index=True)
    description: Optional[str] = Field(sa_column=Column(Text))
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT, index=True)
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    quality_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    created_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    
    # Relationships
    created_by_agent: Agent = Relationship(back_populates="projects")
    artist_profiles: List["ArtistProfile"] = Relationship(back_populates="project")
    song_analyses: List["SongAnalysis"] = Relationship(back_populates="project")
    visual_concepts: List["VisualConcept"] = Relationship(back_populates="project")
    video_segments: List["VideoSegment"] = Relationship(back_populates="project")
    video_sequences: List["VideoSequence"] = Relationship(back_populates="project")
    agent_activities: List["AgentActivity"] = Relationship(back_populates="project")
    workflow_executions: List["WorkflowExecution"] = Relationship(back_populates="project")
    quality_assessments: List["QualityAssessment"] = Relationship(back_populates="project")
    knowledge_entries: List["KnowledgeEntry"] = Relationship(back_populates="project")
    shared_resources: List["SharedResource"] = Relationship(back_populates="project")

# Music Research Models
class ArtistProfile(EmpireBaseModel, table=True):
    __tablename__ = "artist_profiles"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    name: str = Field(index=True)
    genres: List[str] = Field(sa_column=Column(ARRAY(String)))
    audio_features: Dict[str, Any] = Field(sa_column=Column(JSON))
    biography: Optional[str] = Field(sa_column=Column(Text))
    similar_artists: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String))
    )
    cultural_context: Optional[str] = Field(sa_column=Column(Text))
    created_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    
    # Relationships
    project: Project = Relationship(back_populates="artist_profiles")
    created_by_agent: Agent = Relationship(back_populates="artist_profiles")
    song_analyses: List["SongAnalysis"] = Relationship(back_populates="artist_profile")

class SongAnalysis(EmpireBaseModel, table=True):
    __tablename__ = "song_analyses"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    artist_profile_id: Optional[uuid.UUID] = Field(
        foreign_key="artist_profiles.id",
        index=True
    )
    title: str = Field(index=True)
    artist_name: str = Field(index=True)
    lyrics: Optional[str] = Field(sa_column=Column(Text))
    themes: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String))
    )
    mood: Optional[str] = None
    musical_characteristics: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    audio_features: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    annotations: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    created_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    
    # Relationships
    project: Project = Relationship(back_populates="song_analyses")
    artist_profile: Optional[ArtistProfile] = Relationship(back_populates="song_analyses")
    created_by_agent: Agent = Relationship(back_populates="song_analyses")

# Visual Concept Models
class VisualConcept(EmpireBaseModel, table=True):
    __tablename__ = "visual_concepts"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    name: str = Field(index=True)
    description: str = Field(sa_column=Column(Text))
    style: str
    mood: str
    color_palette: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    style_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    created_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    
    # Relationships
    project: Project = Relationship(back_populates="visual_concepts")
    created_by_agent: Agent = Relationship(back_populates="visual_concepts")
    reference_images: List["ReferenceImage"] = Relationship(back_populates="visual_concept")
    video_segments: List["VideoSegment"] = Relationship(back_populates="visual_concept")

class ReferenceImage(EmpireBaseModel, table=True):
    __tablename__ = "reference_images"
    
    visual_concept_id: uuid.UUID = Field(foreign_key="visual_concepts.id", index=True)
    url: str
    source: str
    description: Optional[str] = Field(sa_column=Column(Text))
    image_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    image_type: str = Field(index=True)  # "reference", "generated", "inspiration"
    
    # Relationships
    visual_concept: VisualConcept = Relationship(back_populates="reference_images")

# Video Production Models
class VideoSegment(EmpireBaseModel, table=True):
    __tablename__ = "video_segments"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    visual_concept_id: Optional[uuid.UUID] = Field(
        foreign_key="visual_concepts.id",
        index=True
    )
    name: str = Field(index=True)
    description: str = Field(sa_column=Column(Text))
    duration_seconds: int
    style: str
    operation_id: Optional[str] = Field(index=True)
    status: ActivityStatus = Field(default=ActivityStatus.PENDING, index=True)
    video_path: Optional[str] = None
    generation_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    quality_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    created_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    
    # Relationships
    project: Project = Relationship(back_populates="video_segments")
    visual_concept: Optional[VisualConcept] = Relationship(back_populates="video_segments")
    created_by_agent: Agent = Relationship(back_populates="video_segments")

class VideoSequence(EmpireBaseModel, table=True):
    __tablename__ = "video_sequences"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    name: str = Field(index=True)
    description: str = Field(sa_column=Column(Text))
    total_duration: int
    segment_order: List[str] = Field(sa_column=Column(ARRAY(String)))
    final_video_path: Optional[str] = None
    status: ActivityStatus = Field(default=ActivityStatus.PENDING, index=True)
    completed_at: Optional[datetime] = None
    
    # Relationships
    project: Project = Relationship(back_populates="video_sequences")

# Workflow & Activity Models
class AgentActivity(EmpireBaseModel, table=True):
    __tablename__ = "agent_activities"
    
    agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    operation: str = Field(index=True)
    description: str = Field(sa_column=Column(Text))
    status: ActivityStatus = Field(default=ActivityStatus.PENDING, index=True)
    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON)
    )
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    correlation_id: Optional[str] = Field(index=True)
    
    # Relationships
    agent: Agent = Relationship(back_populates="activities")
    project: Project = Relationship(back_populates="agent_activities")

class WorkflowExecution(EmpireBaseModel, table=True):
    __tablename__ = "workflow_executions"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    workflow_pattern: WorkflowPattern = Field(index=True)
    status: ActivityStatus = Field(default=ActivityStatus.PENDING, index=True)
    step_results: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    error_log: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String))
    )
    total_execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    initiated_by: str
    
    # Relationships
    project: Project = Relationship(back_populates="workflow_executions")

# Quality Control Models
class QualityAssessment(EmpireBaseModel, table=True):
    __tablename__ = "quality_assessments"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    content_type: ContentType = Field(index=True)
    content_id: uuid.UUID = Field(index=True)
    quality_scores: Dict[str, Any] = Field(sa_column=Column(JSON))
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool = Field(index=True)
    issues: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String))
    )
    recommendations: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String))
    )
    assessed_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"server_default": func.now()}
    )
    assessed_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    
    # Relationships
    project: Project = Relationship(back_populates="quality_assessments")
    assessed_by_agent: Agent = Relationship(back_populates="quality_assessments")

# Knowledge Base Models
class KnowledgeEntry(EmpireBaseModel, table=True):
    __tablename__ = "knowledge_entries"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    entry_type: EntryType = Field(index=True)
    title: str = Field(index=True)
    content: str = Field(sa_column=Column(Text))
    structured_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    tags: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String))
    )
    created_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    updated_by_agent_id: Optional[uuid.UUID] = Field(
        foreign_key="agents.id",
        index=True
    )
    
    # Relationships
    project: Project = Relationship(back_populates="knowledge_entries")
    created_by_agent: Agent = Relationship(
        back_populates="knowledge_entries_created",
        sa_relationship_kwargs={"foreign_keys": "[KnowledgeEntry.created_by_agent_id]"}
    )
    updated_by_agent: Optional[Agent] = Relationship(
        back_populates="knowledge_entries_updated",
        sa_relationship_kwargs={"foreign_keys": "[KnowledgeEntry.updated_by_agent_id]"}
    )

class SharedResource(EmpireBaseModel, table=True):
    __tablename__ = "shared_resources"
    
    project_id: uuid.UUID = Field(foreign_key="projects.id", index=True)
    name: str = Field(index=True)
    resource_type: ResourceType = Field(index=True)
    file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    access_level: AccessLevel = Field(default=AccessLevel.PROJECT, index=True)
    created_by_agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    
    # Relationships
    project: Project = Relationship(back_populates="shared_resources")
    created_by_agent: Agent = Relationship(back_populates="shared_resources")
```

## Repository Pattern Implementation

### Generic Repository Base

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any
from sqlmodel import Session, select, and_, or_
from sqlalchemy import desc, asc

SQLModelType = TypeVar("SQLModelType", bound=SQLModel)

class GenericRepository(Generic[SQLModelType], ABC):
    """Abstract base repository for CRUD operations."""
    
    @abstractmethod
    async def get_by_id(self, id: uuid.UUID) -> Optional[SQLModelType]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        **filters
    ) -> List[SQLModelType]:
        """List entities with filtering and pagination."""
        pass
    
    @abstractmethod
    async def add(self, entity: SQLModelType) -> SQLModelType:
        """Add new entity."""
        pass
    
    @abstractmethod
    async def update(self, entity: SQLModelType) -> SQLModelType:
        """Update existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, id: uuid.UUID) -> bool:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    async def count(self, **filters) -> int:
        """Count entities matching filters."""
        pass

class GenericSQLRepository(GenericRepository[SQLModelType]):
    """Generic SQLModel repository implementation."""
    
    def __init__(self, session: Session, model: type[SQLModelType]):
        self.session = session
        self.model = model
    
    async def get_by_id(self, id: uuid.UUID) -> Optional[SQLModelType]:
        return await self.session.get(self.model, id)
    
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        **filters
    ) -> List[SQLModelType]:
        statement = select(self.model)
        
        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model, key) and value is not None:
                statement = statement.where(getattr(self.model, key) == value)
        
        # Apply ordering
        if order_by:
            if order_by.startswith("-"):
                field = order_by[1:]
                if hasattr(self.model, field):
                    statement = statement.order_by(desc(getattr(self.model, field)))
            else:
                if hasattr(self.model, order_by):
                    statement = statement.order_by(asc(getattr(self.model, order_by)))
        
        # Apply pagination
        statement = statement.offset(offset).limit(limit)
        
        result = await self.session.exec(statement)
        return result.all()
    
    async def add(self, entity: SQLModelType) -> SQLModelType:
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: SQLModelType) -> SQLModelType:
        entity.updated_at = datetime.utcnow()
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: uuid.UUID) -> bool:
        entity = await self.get_by_id(id)
        if entity:
            await self.session.delete(entity)
            await self.session.commit()
            return True
        return False
    
    async def count(self, **filters) -> int:
        statement = select(func.count(self.model.id))
        
        for key, value in filters.items():
            if hasattr(self.model, key) and value is not None:
                statement = statement.where(getattr(self.model, key) == value)
        
        result = await self.session.exec(statement)
        return result.one()
```

### Specialized Repositories

```python
# Project Repository
class ProjectRepositoryBase(GenericRepository[Project]):
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Project]:
        pass
    
    @abstractmethod
    async def get_active_projects(self) -> List[Project]:
        pass
    
    @abstractmethod
    async def get_projects_by_agent(self, agent_id: uuid.UUID) -> List[Project]:
        pass

class ProjectRepository(GenericSQLRepository[Project], ProjectRepositoryBase):
    async def get_by_name(self, name: str) -> Optional[Project]:
        statement = select(Project).where(Project.name == name)
        result = await self.session.exec(statement)
        return result.first()
    
    async def get_active_projects(self) -> List[Project]:
        statement = select(Project).where(
            Project.status.in_([
                ProjectStatus.ACTIVE,
                ProjectStatus.IN_PROGRESS
            ])
        )
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_projects_by_agent(self, agent_id: uuid.UUID) -> List[Project]:
        statement = select(Project).where(Project.created_by_agent_id == agent_id)
        result = await self.session.exec(statement)
        return result.all()

# Agent Repository
class AgentRepositoryBase(GenericRepository[Agent]):
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Agent]:
        pass
    
    @abstractmethod
    async def get_by_type(self, agent_type: AgentType) -> List[Agent]:
        pass
    
    @abstractmethod
    async def get_active_agents(self) -> List[Agent]:
        pass
    
    @abstractmethod
    async def update_status(self, agent_id: uuid.UUID, status: AgentStatus) -> bool:
        pass

class AgentRepository(GenericSQLRepository[Agent], AgentRepositoryBase):
    async def get_by_name(self, name: str) -> Optional[Agent]:
        statement = select(Agent).where(Agent.name == name)
        result = await self.session.exec(statement)
        return result.first()
    
    async def get_by_type(self, agent_type: AgentType) -> List[Agent]:
        statement = select(Agent).where(Agent.agent_type == agent_type)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_active_agents(self) -> List[Agent]:
        statement = select(Agent).where(Agent.status == AgentStatus.ACTIVE)
        result = await self.session.exec(statement)
        return result.all()
    
    async def update_status(self, agent_id: uuid.UUID, status: AgentStatus) -> bool:
        agent = await self.get_by_id(agent_id)
        if agent:
            agent.status = status
            agent.last_active = datetime.utcnow()
            await self.update(agent)
            return True
        return False

# Music Research Repositories
class ArtistProfileRepositoryBase(GenericRepository[ArtistProfile]):
    @abstractmethod
    async def get_by_project(self, project_id: uuid.UUID) -> List[ArtistProfile]:
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> List[ArtistProfile]:
        pass
    
    @abstractmethod
    async def search_by_genre(self, genre: str) -> List[ArtistProfile]:
        pass

class ArtistProfileRepository(GenericSQLRepository[ArtistProfile], ArtistProfileRepositoryBase):
    async def get_by_project(self, project_id: uuid.UUID) -> List[ArtistProfile]:
        statement = select(ArtistProfile).where(ArtistProfile.project_id == project_id)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_by_name(self, name: str) -> List[ArtistProfile]:
        statement = select(ArtistProfile).where(
            ArtistProfile.name.ilike(f"%{name}%")
        )
        result = await self.session.exec(statement)
        return result.all()
    
    async def search_by_genre(self, genre: str) -> List[ArtistProfile]:
        statement = select(ArtistProfile).where(
            ArtistProfile.genres.contains([genre])
        )
        result = await self.session.exec(statement)
        return result.all()

class SongAnalysisRepositoryBase(GenericRepository[SongAnalysis]):
    @abstractmethod
    async def get_by_project(self, project_id: uuid.UUID) -> List[SongAnalysis]:
        pass
    
    @abstractmethod
    async def get_by_artist_profile(self, artist_profile_id: uuid.UUID) -> List[SongAnalysis]:
        pass
    
    @abstractmethod
    async def search_by_title(self, title: str) -> List[SongAnalysis]:
        pass
    
    @abstractmethod
    async def search_by_mood(self, mood: str) -> List[SongAnalysis]:
        pass

class SongAnalysisRepository(GenericSQLRepository[SongAnalysis], SongAnalysisRepositoryBase):
    async def get_by_project(self, project_id: uuid.UUID) -> List[SongAnalysis]:
        statement = select(SongAnalysis).where(SongAnalysis.project_id == project_id)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_by_artist_profile(self, artist_profile_id: uuid.UUID) -> List[SongAnalysis]:
        statement = select(SongAnalysis).where(
            SongAnalysis.artist_profile_id == artist_profile_id
        )
        result = await self.session.exec(statement)
        return result.all()
    
    async def search_by_title(self, title: str) -> List[SongAnalysis]:
        statement = select(SongAnalysis).where(
            SongAnalysis.title.ilike(f"%{title}%")
        )
        result = await self.session.exec(statement)
        return result.all()
    
    async def search_by_mood(self, mood: str) -> List[SongAnalysis]:
        statement = select(SongAnalysis).where(SongAnalysis.mood == mood)
        result = await self.session.exec(statement)
        return result.all()

# Visual Concept Repositories
class VisualConceptRepositoryBase(GenericRepository[VisualConcept]):
    @abstractmethod
    async def get_by_project(self, project_id: uuid.UUID) -> List[VisualConcept]:
        pass
    
    @abstractmethod
    async def get_by_style(self, style: str) -> List[VisualConcept]:
        pass
    
    @abstractmethod
    async def get_high_confidence(self, min_score: float = 0.8) -> List[VisualConcept]:
        pass

class VisualConceptRepository(GenericSQLRepository[VisualConcept], VisualConceptRepositoryBase):
    async def get_by_project(self, project_id: uuid.UUID) -> List[VisualConcept]:
        statement = select(VisualConcept).where(VisualConcept.project_id == project_id)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_by_style(self, style: str) -> List[VisualConcept]:
        statement = select(VisualConcept).where(VisualConcept.style == style)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_high_confidence(self, min_score: float = 0.8) -> List[VisualConcept]:
        statement = select(VisualConcept).where(
            VisualConcept.confidence_score >= min_score
        )
        result = await self.session.exec(statement)
        return result.all()

# Video Production Repositories
class VideoSegmentRepositoryBase(GenericRepository[VideoSegment]):
    @abstractmethod
    async def get_by_project(self, project_id: uuid.UUID) -> List[VideoSegment]:
        pass
    
    @abstractmethod
    async def get_by_status(self, status: ActivityStatus) -> List[VideoSegment]:
        pass
    
    @abstractmethod
    async def get_by_visual_concept(self, visual_concept_id: uuid.UUID) -> List[VideoSegment]:
        pass
    
    @abstractmethod
    async def get_completed_segments(self, project_id: uuid.UUID) -> List[VideoSegment]:
        pass

class VideoSegmentRepository(GenericSQLRepository[VideoSegment], VideoSegmentRepositoryBase):
    async def get_by_project(self, project_id: uuid.UUID) -> List[VideoSegment]:
        statement = select(VideoSegment).where(VideoSegment.project_id == project_id)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_by_status(self, status: ActivityStatus) -> List[VideoSegment]:
        statement = select(VideoSegment).where(VideoSegment.status == status)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_by_visual_concept(self, visual_concept_id: uuid.UUID) -> List[VideoSegment]:
        statement = select(VideoSegment).where(
            VideoSegment.visual_concept_id == visual_concept_id
        )
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_completed_segments(self, project_id: uuid.UUID) -> List[VideoSegment]:
        statement = select(VideoSegment).where(
            and_(
                VideoSegment.project_id == project_id,
                VideoSegment.status == ActivityStatus.COMPLETED
            )
        )
        result = await self.session.exec(statement)
        return result.all()

# Knowledge Base Repositories
class KnowledgeEntryRepositoryBase(GenericRepository[KnowledgeEntry]):
    @abstractmethod
    async def get_by_project(self, project_id: uuid.UUID) -> List[KnowledgeEntry]:
        pass
    
    @abstractmethod
    async def get_by_type(self, entry_type: EntryType) -> List[KnowledgeEntry]:
        pass
    
    @abstractmethod
    async def search_by_tags(self, tags: List[str]) -> List[KnowledgeEntry]:
        pass
    
    @abstractmethod
    async def search_content(self, query: str) -> List[KnowledgeEntry]:
        pass

class KnowledgeEntryRepository(GenericSQLRepository[KnowledgeEntry], KnowledgeEntryRepositoryBase):
    async def get_by_project(self, project_id: uuid.UUID) -> List[KnowledgeEntry]:
        statement = select(KnowledgeEntry).where(KnowledgeEntry.project_id == project_id)
        result = await self.session.exec(statement)
        return result.all()
    
    async def get_by_type(self, entry_type: EntryType) -> List[KnowledgeEntry]:
        statement = select(KnowledgeEntry).where(KnowledgeEntry.entry_type == entry_type)
        result = await self.session.exec(statement)
        return result.all()
    
    async def search_by_tags(self, tags: List[str]) -> List[KnowledgeEntry]:
        statement = select(KnowledgeEntry)
        for tag in tags:
            statement = statement.where(KnowledgeEntry.tags.contains([tag]))
        result = await self.session.exec(statement)
        return result.all()
    
    async def search_content(self, query: str) -> List[KnowledgeEntry]:
        statement = select(KnowledgeEntry).where(
            or_(
                KnowledgeEntry.title.ilike(f"%{query}%"),
                KnowledgeEntry.content.ilike(f"%{query}%")
            )
        )
        result = await self.session.exec(statement)
        return result.all()
```

## Unit of Work Pattern

```python
from abc import ABC, abstractmethod
from typing import Callable
from sqlmodel import Session
from contextlib import asynccontextmanager

class UnitOfWorkBase(ABC):
    """Abstract Unit of Work for transaction management."""
    
    # Repository properties
    projects: ProjectRepositoryBase
    agents: AgentRepositoryBase
    artist_profiles: ArtistProfileRepositoryBase
    song_analyses: SongAnalysisRepositoryBase
    visual_concepts: VisualConceptRepositoryBase
    video_segments: VideoSegmentRepositoryBase
    video_sequences: GenericRepository[VideoSequence]
    agent_activities: GenericRepository[AgentActivity]
    workflow_executions: GenericRepository[WorkflowExecution]
    quality_assessments: GenericRepository[QualityAssessment]
    knowledge_entries: KnowledgeEntryRepositoryBase
    shared_resources: GenericRepository[SharedResource]
    reference_images: GenericRepository[ReferenceImage]
    
    @abstractmethod
    async def __aenter__(self):
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @abstractmethod
    async def commit(self):
        pass
    
    @abstractmethod
    async def rollback(self):
        pass

class SQLModelUnitOfWork(UnitOfWorkBase):
    """SQLModel implementation of Unit of Work pattern."""
    
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        self.session: Optional[Session] = None
    
    async def __aenter__(self):
        self.session = self.session_factory()
        
        # Initialize all repositories with the same session
        self.projects = ProjectRepository(self.session, Project)
        self.agents = AgentRepository(self.session, Agent)
        self.artist_profiles = ArtistProfileRepository(self.session, ArtistProfile)
        self.song_analyses = SongAnalysisRepository(self.session, SongAnalysis)
        self.visual_concepts = VisualConceptRepository(self.session, VisualConcept)
        self.video_segments = VideoSegmentRepository(self.session, VideoSegment)
        self.video_sequences = GenericSQLRepository(self.session, VideoSequence)
        self.agent_activities = GenericSQLRepository(self.session, AgentActivity)
        self.workflow_executions = GenericSQLRepository(self.session, WorkflowExecution)
        self.quality_assessments = GenericSQLRepository(self.session, QualityAssessment)
        self.knowledge_entries = KnowledgeEntryRepository(self.session, KnowledgeEntry)
        self.shared_resources = GenericSQLRepository(self.session, SharedResource)
        self.reference_images = GenericSQLRepository(self.session, ReferenceImage)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
        
        if self.session:
            await self.session.close()
    
    async def commit(self):
        if self.session:
            await self.session.commit()
    
    async def rollback(self):
        if self.session:
            await self.session.rollback()
```

## Database Configuration

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
import os

# Database Configuration
class DatabaseConfig:
    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://empire:empire@localhost:5432/empire_ai"
        )
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    
    def create_engine(self):
        return create_async_engine(
            self.database_url,
            echo=self.echo,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_pre_ping=True
        )

# Database Setup
config = DatabaseConfig()
engine = config.create_engine()

# Session Factory
async_session_factory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create Tables
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

# Database Initialization
async def init_database():
    """Initialize database with required extensions and tables."""
    # Enable UUID extension
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";"))
    
    # Create all tables
    await create_tables()
```

## CRUD Service Layer

```python
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class EmpireAIService:
    """High-level service for Empire.AI operations."""
    
    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self.session_factory = session_factory
    
    # Project Management
    async def create_project(
        self,
        name: str,
        description: str,
        agent_id: uuid.UUID,
        user_preferences: Optional[Dict[str, Any]] = None,
        quality_requirements: Optional[Dict[str, Any]] = None
    ) -> Project:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            project = Project(
                name=name,
                description=description,
                created_by_agent_id=agent_id,
                user_preferences=user_preferences or {},
                quality_requirements=quality_requirements or {}
            )
            return await uow.projects.add(project)
    
    async def get_project(self, project_id: uuid.UUID) -> Optional[Project]:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            return await uow.projects.get_by_id(project_id)
    
    async def update_project_status(
        self,
        project_id: uuid.UUID,
        status: ProjectStatus
    ) -> Optional[Project]:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            project = await uow.projects.get_by_id(project_id)
            if project:
                project.status = status
                return await uow.projects.update(project)
            return None
    
    # Music Research Operations
    async def save_artist_profile(
        self,
        project_id: uuid.UUID,
        agent_id: uuid.UUID,
        name: str,
        genres: List[str],
        audio_features: Dict[str, Any],
        biography: Optional[str] = None,
        similar_artists: Optional[List[str]] = None,
        cultural_context: Optional[str] = None
    ) -> ArtistProfile:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            profile = ArtistProfile(
                project_id=project_id,
                created_by_agent_id=agent_id,
                name=name,
                genres=genres,
                audio_features=audio_features,
                biography=biography,
                similar_artists=similar_artists or [],
                cultural_context=cultural_context
            )
            return await uow.artist_profiles.add(profile)
    
    async def save_song_analysis(
        self,
        project_id: uuid.UUID,
        agent_id: uuid.UUID,
        title: str,
        artist_name: str,
        artist_profile_id: Optional[uuid.UUID] = None,
        lyrics: Optional[str] = None,
        themes: Optional[List[str]] = None,
        mood: Optional[str] = None,
        musical_characteristics: Optional[Dict[str, Any]] = None,
        audio_features: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None
    ) -> SongAnalysis:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            analysis = SongAnalysis(
                project_id=project_id,
                created_by_agent_id=agent_id,
                artist_profile_id=artist_profile_id,
                title=title,
                artist_name=artist_name,
                lyrics=lyrics,
                themes=themes or [],
                mood=mood,
                musical_characteristics=musical_characteristics or {},
                audio_features=audio_features or {},
                annotations=annotations or {}
            )
            return await uow.song_analyses.add(analysis)
    
    # Visual Concept Operations
    async def save_visual_concept(
        self,
        project_id: uuid.UUID,
        agent_id: uuid.UUID,
        name: str,
        description: str,
        style: str,
        mood: str,
        confidence_score: float,
        color_palette: Optional[str] = None,
        style_parameters: Optional[Dict[str, Any]] = None
    ) -> VisualConcept:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            concept = VisualConcept(
                project_id=project_id,
                created_by_agent_id=agent_id,
                name=name,
                description=description,
                style=style,
                mood=mood,
                confidence_score=confidence_score,
                color_palette=color_palette,
                style_parameters=style_parameters or {}
            )
            return await uow.visual_concepts.add(concept)
    
    # Video Production Operations
    async def save_video_segment(
        self,
        project_id: uuid.UUID,
        agent_id: uuid.UUID,
        name: str,
        description: str,
        duration_seconds: int,
        style: str,
        visual_concept_id: Optional[uuid.UUID] = None,
        operation_id: Optional[str] = None,
        generation_parameters: Optional[Dict[str, Any]] = None
    ) -> VideoSegment:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            segment = VideoSegment(
                project_id=project_id,
                created_by_agent_id=agent_id,
                visual_concept_id=visual_concept_id,
                name=name,
                description=description,
                duration_seconds=duration_seconds,
                style=style,
                operation_id=operation_id,
                generation_parameters=generation_parameters or {}
            )
            return await uow.video_segments.add(segment)
    
    # Knowledge Base Operations
    async def add_knowledge_entry(
        self,
        project_id: uuid.UUID,
        agent_id: uuid.UUID,
        entry_type: EntryType,
        title: str,
        content: str,
        structured_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> KnowledgeEntry:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            entry = KnowledgeEntry(
                project_id=project_id,
                created_by_agent_id=agent_id,
                entry_type=entry_type,
                title=title,
                content=content,
                structured_data=structured_data or {},
                tags=tags or []
            )
            return await uow.knowledge_entries.add(entry)
    
    async def search_knowledge(
        self,
        project_id: uuid.UUID,
        query: Optional[str] = None,
        entry_type: Optional[EntryType] = None,
        tags: Optional[List[str]] = None
    ) -> List[KnowledgeEntry]:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            if query:
                return await uow.knowledge_entries.search_content(query)
            elif entry_type:
                return await uow.knowledge_entries.get_by_type(entry_type)
            elif tags:
                return await uow.knowledge_entries.search_by_tags(tags)
            else:
                return await uow.knowledge_entries.get_by_project(project_id)
    
    # Agent Activity Tracking
    async def log_agent_activity(
        self,
        agent_id: uuid.UUID,
        project_id: uuid.UUID,
        operation: str,
        description: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        status: ActivityStatus = ActivityStatus.COMPLETED,
        error_details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> AgentActivity:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            activity = AgentActivity(
                agent_id=agent_id,
                project_id=project_id,
                operation=operation,
                description=description,
                status=status,
                input_data=input_data,
                output_data=output_data,
                error_details=error_details,
                execution_time=execution_time,
                correlation_id=correlation_id,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            return await uow.agent_activities.add(activity)
    
    # Project Analytics
    async def get_project_summary(self, project_id: uuid.UUID) -> Dict[str, Any]:
        async with SQLModelUnitOfWork(self.session_factory) as uow:
            project = await uow.projects.get_by_id(project_id)
            if not project:
                return {}
            
            artist_profiles = await uow.artist_profiles.get_by_project(project_id)
            song_analyses = await uow.song_analyses.get_by_project(project_id)
            visual_concepts = await uow.visual_concepts.get_by_project(project_id)
            video_segments = await uow.video_segments.get_by_project(project_id)
            knowledge_entries = await uow.knowledge_entries.get_by_project(project_id)
            
            return {
                "project": project,
                "stats": {
                    "artist_profiles": len(artist_profiles),
                    "song_analyses": len(song_analyses),
                    "visual_concepts": len(visual_concepts),
                    "video_segments": len(video_segments),
                    "knowledge_entries": len(knowledge_entries)
                },
                "latest_activity": datetime.utcnow()  # Could be computed from activities
            }
```

## Database Indexing Strategy

```sql
-- Performance Indexes for PostgreSQL

-- Project indexes
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_created_by_agent ON projects(created_by_agent_id);
CREATE INDEX idx_projects_created_at ON projects(created_at);

-- Agent indexes
CREATE INDEX idx_agents_type ON agents(agent_type);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_name ON agents(name);
CREATE INDEX idx_agents_last_active ON agents(last_active);

-- Artist profile indexes
CREATE INDEX idx_artist_profiles_project ON artist_profiles(project_id);
CREATE INDEX idx_artist_profiles_name ON artist_profiles(name);
CREATE INDEX idx_artist_profiles_created_by ON artist_profiles(created_by_agent_id);
CREATE INDEX idx_artist_profiles_genres ON artist_profiles USING GIN(genres);

-- Song analysis indexes
CREATE INDEX idx_song_analyses_project ON song_analyses(project_id);
CREATE INDEX idx_song_analyses_artist_profile ON song_analyses(artist_profile_id);
CREATE INDEX idx_song_analyses_title ON song_analyses(title);
CREATE INDEX idx_song_analyses_artist_name ON song_analyses(artist_name);
CREATE INDEX idx_song_analyses_mood ON song_analyses(mood);
CREATE INDEX idx_song_analyses_themes ON song_analyses USING GIN(themes);

-- Visual concept indexes
CREATE INDEX idx_visual_concepts_project ON visual_concepts(project_id);
CREATE INDEX idx_visual_concepts_style ON visual_concepts(style);
CREATE INDEX idx_visual_concepts_confidence ON visual_concepts(confidence_score);
CREATE INDEX idx_visual_concepts_created_by ON visual_concepts(created_by_agent_id);

-- Video segment indexes
CREATE INDEX idx_video_segments_project ON video_segments(project_id);
CREATE INDEX idx_video_segments_visual_concept ON video_segments(visual_concept_id);
CREATE INDEX idx_video_segments_status ON video_segments(status);
CREATE INDEX idx_video_segments_operation ON video_segments(operation_id);
CREATE INDEX idx_video_segments_created_by ON video_segments(created_by_agent_id);

-- Agent activity indexes
CREATE INDEX idx_agent_activities_agent ON agent_activities(agent_id);
CREATE INDEX idx_agent_activities_project ON agent_activities(project_id);
CREATE INDEX idx_agent_activities_operation ON agent_activities(operation);
CREATE INDEX idx_agent_activities_status ON agent_activities(status);
CREATE INDEX idx_agent_activities_correlation ON agent_activities(correlation_id);
CREATE INDEX idx_agent_activities_started_at ON agent_activities(started_at);

-- Quality assessment indexes
CREATE INDEX idx_quality_assessments_project ON quality_assessments(project_id);
CREATE INDEX idx_quality_assessments_content_type ON quality_assessments(content_type);
CREATE INDEX idx_quality_assessments_content_id ON quality_assessments(content_id);
CREATE INDEX idx_quality_assessments_passed ON quality_assessments(passed);
CREATE INDEX idx_quality_assessments_assessed_at ON quality_assessments(assessed_at);

-- Knowledge entry indexes
CREATE INDEX idx_knowledge_entries_project ON knowledge_entries(project_id);
CREATE INDEX idx_knowledge_entries_type ON knowledge_entries(entry_type);
CREATE INDEX idx_knowledge_entries_tags ON knowledge_entries USING GIN(tags);
CREATE INDEX idx_knowledge_entries_title ON knowledge_entries(title);
CREATE INDEX idx_knowledge_entries_content_fts ON knowledge_entries USING GIN(to_tsvector('english', content));

-- Shared resource indexes
CREATE INDEX idx_shared_resources_project ON shared_resources(project_id);
CREATE INDEX idx_shared_resources_type ON shared_resources(resource_type);
CREATE INDEX idx_shared_resources_access_level ON shared_resources(access_level);

-- Composite indexes for common queries
CREATE INDEX idx_agent_activities_project_agent ON agent_activities(project_id, agent_id);
CREATE INDEX idx_visual_concepts_project_style ON visual_concepts(project_id, style);
CREATE INDEX idx_video_segments_project_status ON video_segments(project_id, status);
CREATE INDEX idx_knowledge_entries_project_type ON knowledge_entries(project_id, entry_type);
```

## Migration Strategy

```python
# Alembic migration for initial schema
"""
Initial Empire.AI schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-11-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

def upgrade():
    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    
    # Create all tables using SQLModel metadata
    # This would be generated automatically by Alembic
    pass

def downgrade():
    # Drop all tables in reverse order
    pass
```

## Usage Examples

```python
# Initialize service
service = EmpireAIService(async_session_factory)

# Create a project
project = await service.create_project(
    name="Taylor Swift - Anti-Hero Music Video",
    description="Create a music video for Anti-Hero by Taylor Swift",
    agent_id=orchestrator_agent_id,
    user_preferences={
        "style": "cinematic",
        "mood": "introspective",
        "duration": 240
    },
    quality_requirements={
        "min_resolution": "1080p",
        "frame_rate": 24,
        "quality_score_threshold": 0.8
    }
)

# Save music research data
artist_profile = await service.save_artist_profile(
    project_id=project.id,
    agent_id=music_research_agent_id,
    name="Taylor Swift",
    genres=["pop", "country", "indie"],
    audio_features={
        "energy": 0.7,
        "valence": 0.4,
        "danceability": 0.6
    },
    biography="American singer-songwriter...",
    similar_artists=["Lorde", "Lana Del Rey", "Phoebe Bridgers"]
)

song_analysis = await service.save_song_analysis(
    project_id=project.id,
    agent_id=music_research_agent_id,
    artist_profile_id=artist_profile.id,
    title="Anti-Hero",
    artist_name="Taylor Swift",
    lyrics="I have this thing where I get older...",
    themes=["self-reflection", "anxiety", "fame"],
    mood="introspective",
    musical_characteristics={
        "tempo": 97,
        "key": "C major",
        "time_signature": "4/4"
    }
)

# Save visual concepts
visual_concept = await service.save_visual_concept(
    project_id=project.id,
    agent_id=visual_concept_agent_id,
    name="Mirror Reflection Concept",
    description="Dark, moody visuals with mirror imagery reflecting inner thoughts",
    style="cinematic",
    mood="introspective",
    confidence_score=0.85,
    color_palette="dark blues, muted grays, warm highlights",
    style_parameters={
        "lighting": "low-key",
        "camera_movement": "slow",
        "color_grading": "desaturated"
    }
)

# Save video segments
video_segment = await service.save_video_segment(
    project_id=project.id,
    agent_id=video_production_agent_id,
    visual_concept_id=visual_concept.id,
    name="Opening Mirror Scene",
    description="Taylor looking at herself in a bathroom mirror",
    duration_seconds=15,
    style="cinematic",
    generation_parameters={
        "prompt": "Taylor Swift looking in bathroom mirror, moody lighting",
        "style": "cinematic",
        "duration": 15
    }
)

# Add knowledge entries
await service.add_knowledge_entry(
    project_id=project.id,
    agent_id=orchestrator_agent_id,
    entry_type=EntryType.DECISION_LOG,
    title="Visual Style Decision",
    content="Chose cinematic style based on song's introspective nature and artist's recent aesthetic evolution",
    structured_data={
        "decision_factors": [
            "Song mood analysis",
            "Artist brand consistency",
            "Target audience preferences"
        ],
        "alternatives_considered": ["animated", "performance-based", "abstract"]
    },
    tags=["visual-style", "decision", "concept-development"]
)

# Get project summary
summary = await service.get_project_summary(project.id)
print(f"Project has {summary['stats']['visual_concepts']} visual concepts")

# Search knowledge base
knowledge_results = await service.search_knowledge(
    project_id=project.id,
    tags=["visual-style"]
)
```

This comprehensive database design provides:

✅ **Complete Schema** - All entities for multi-agent workflows  
✅ **Type Safety** - Full SQLModel integration with Pydantic validation  
✅ **Repository Pattern** - Clean separation of data access logic  
✅ **Unit of Work** - Transaction management across repositories  
✅ **Performance Optimized** - Strategic indexing for common queries  
✅ **Knowledge Base** - Comprehensive storage for agent insights  
✅ **Audit Trail** - Complete tracking of all agent activities  
✅ **Shared Workspace** - Resource sharing between agents  
✅ **Quality Control** - Assessment tracking and metrics  
✅ **Scalable Architecture** - Supports complex multi-agent coordination  

The design supports all our agent coordination patterns while providing a robust foundation for the Empire.AI knowledge base and shared workspace!
