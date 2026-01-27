from typing import List, Optional
from pydantic import BaseModel, Field

# 1. 定义更细粒度的实体类型
class Entity(BaseModel):
    name: str = Field(description="实体名称，例如：'象鼻山'、'100路公交车'、'夜游两江四湖船票'")
    type: str = Field(description="实体类型。请从以下选项中选择：[景点, 地点, 交通工具, 票务/价格, 规则/政策, 活动/演艺, 设施, 组织/公司]")
    description: str = Field(description="实体的详细属性或描述。例如票价的具体金额、公交车的运营时间等。", default="")

# 2. 定义更明确的关系类型
class Relationship(BaseModel):
    source: str = Field(description="源实体名称")
    target: str = Field(description="目标实体名称")
    relation_type: str = Field(description="关系类型。请从以下选项中选择：[LOCATED_IN(位于), NEAR(毗邻), COSTS(价格为), CONTAINS(包含), ROUTE_TO(通往/到达), REQUIRES(要求), APPLIES_TO(适用于), HAS_FACILITY(拥有设施)]")
    description: str = Field(description="关系的补充说明，例如'身高1.2-1.5米适用'或'步行400米到达'", default="")

class GraphResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]