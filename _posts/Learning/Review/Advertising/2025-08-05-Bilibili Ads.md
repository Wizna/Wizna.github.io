![](https://ts1.tc.mm.bing.net/th/id/OIP-C.IIb4LBa6pdsaIDCMNh9WCQHaEx?rs=1&pid=ImgDetMain)


# 背景

本文档展示了在Jekyll博客中使用Mermaid图表的各种示例，包括流程图、思维导图、时序图等。

# 图表示例

## 流程图 (Flowchart)

广告投放决策流程：

<div class="mermaid">
flowchart TD
    A[开始投放策略] --> B{目标用户分析}
    B --> |年轻用户| C[选择B站平台]
    B --> |商务用户| D[选择LinkedIn]
    B --> |大众用户| E[选择微博]
    C --> F[制定内容策略]
    D --> F
    E --> F
    F --> G[投放广告]
    G --> H{效果评估}
    H --> |效果好| I[增加预算]
    H --> |效果差| J[优化策略]
    I --> K[继续投放]
    J --> F
    K --> H
</div>

## 思维导图 (Mindmap)

B站广告策略思维导图：

<div class="mermaid">
mindmap
  root((B站广告策略))
    用户画像
      年龄分布
        Z世代
        千禧一代
      兴趣标签
        二次元
        游戏
        科技
    内容形式
      视频广告
        前贴片
        中插
        后贴片
      信息流广告
      品牌合作
    投放策略
      时间节点
        节假日
        热点事件
      预算分配
        测试预算
        正式投放
      效果监控
        点击率
        转化率
        ROI
</div>

## 时序图 (Sequence Diagram)

广告投放流程时序图：

<div class="mermaid">
sequenceDiagram
    participant 广告主
    participant B站平台
    participant 用户
    participant 数据分析
    
    广告主->>B站平台: 提交广告素材
    B站平台->>B站平台: 审核广告内容
    B站平台->>广告主: 审核结果通知
    广告主->>B站平台: 设置投放参数
    B站平台->>用户: 展示广告
    用户->>B站平台: 用户互动(点击/观看)
    B站平台->>数据分析: 收集数据
    数据分析->>广告主: 效果报告
    广告主->>B站平台: 优化投放策略
</div>

## 甘特图 (Gantt Chart)

广告投放项目时间计划：

<div class="mermaid">
gantt
    title 广告投放项目时间表
    dateFormat  YYYY-MM-DD
    section 准备阶段
    市场调研           :done,    des1, 2025-08-01,2025-08-05
    策略制定           :done,    des2, 2025-08-06, 2025-08-10
    素材制作           :active,  des3, 2025-08-11, 2025-08-15
    section 投放阶段
    测试投放           :         des4, 2025-08-16, 2025-08-20
    正式投放           :         des5, 2025-08-21, 2025-09-20
    section 优化阶段
    数据分析           :         des6, 2025-08-18, 2025-09-22
    策略优化           :         des7, 2025-08-25, 2025-09-25
</div>

## 状态图 (State Diagram)

广告状态流转图：

<div class="mermaid">
stateDiagram-v2
    [*] --> 草稿
    草稿 --> 待审核: 提交审核
    待审核 --> 审核通过: 通过
    待审核 --> 审核拒绝: 拒绝
    审核拒绝 --> 草稿: 修改重提
    审核通过 --> 投放中: 开始投放
    投放中 --> 暂停: 手动暂停
    投放中 --> 完成: 预算用完/时间到期
    暂停 --> 投放中: 恢复投放
    完成 --> [*]
</div>

## 类图 (Class Diagram)

广告系统类结构：

<div class="mermaid">
classDiagram
    class Advertisement {
        +String id
        +String title
        +String content
        +BigDecimal budget
        +Date startTime
        +Date endTime
        +AdStatus status
        +create()
        +update()
        +pause()
        +resume()
    }
    
    class Campaign {
        +String campaignId
        +String name
        +List~Advertisement~ ads
        +addAdvertisement()
        +removeAdvertisement()
        +getTotalBudget()
    }
    
    class User {
        +String userId
        +String name
        +Integer age
        +List~String~ interests
        +getProfile()
    }
    
    class Platform {
        +String platformId
        +String name
        +showAd()
        +collectData()
    }
    
    Campaign ||--o{ Advertisement
    Advertisement }o--|| Platform
    Platform }o--|| User
</div>

## 饼图 (Pie Chart)

广告预算分配：

<div class="mermaid">
pie title 广告预算分配
    "视频广告" : 45
    "信息流广告" : 30
    "品牌合作" : 15
    "其他推广" : 10
</div>

# 使用说明

## 基本语法

在你的markdown文件中使用以下格式：

```html
<div class="mermaid">
图表类型
    图表内容
</div>
```

## 支持的图表类型

1. **flowchart** - 流程图
2. **mindmap** - 思维导图  
3. **sequenceDiagram** - 时序图
4. **gantt** - 甘特图
5. **stateDiagram-v2** - 状态图
6. **classDiagram** - 类图
7. **pie** - 饼图
8. **gitgraph** - Git分支图
9. **erDiagram** - 实体关系图
10. **journey** - 用户旅程图

## 注意事项

- 确保缩进正确，Mermaid对缩进敏感
- 中文内容需要用双引号包围
- 图表渲染需要JavaScript支持
- 建议在本地测试后再发布