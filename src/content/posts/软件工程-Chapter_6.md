---
title: '软件工程 Chapter 6 代码生成'
published: 2026-1-5
description: '软件工程学习记录 Chapter 6 代码生成'
image: ''
tags: ['Software Engineering', 'SE']
category: 'Courses'
draft: false
lang: ''
series: 'SE Courses'
---

# 软件工程 Chapter 6 代码生成

## 第六章核心定位与逻辑脉络

### 核心定位

代码生成是**类设计（第五章）到实际编码的落地环节**，将第五章优化后的“类-属性-方法-关系”模型，转化为可运行的程序代码框架；同时，它是“设计与实现的桥梁”，需确保代码与设计一致，兼顾可维护性、可扩展性（呼应第一章软件工程“工程化、规范化”原则）。

### 核心逻辑链

本章遵循“**工具支撑→规则定义→具体转换→优化封装→架构落地→多视图验证**”的流程：

```
类设计模型（第五章）→ 开发工具（CASE/IDE）→ 变更与逆向工程（一致性保障）→ 单个类代码转换→ 类关系（关联/聚合/组合/依赖）代码实现→ MVC架构代码落地→ 4+1视图验证（非功能需求）→ 支撑最终编码
```

全程围绕“一致性”核心：设计与代码一致、文档与代码一致、需求与代码一致。

## 开发工具与环境：代码生成的基础支撑

代码生成离不开工程化工具，工具的选择与集成直接影响开发效率和设计-代码一致性，是工程化开发的核心保障。

### 1. 核心工具类型与作用

| 工具类型 | 核心定义 | 核心作用（关联第五章类设计） | 示例 |
| :---: | :---: | :---: | :---: |
| CASE工具（计算机辅助软件工程） | 支持软件开发全流程（设计、编码、维护）的工具集，提供设计与代码的双向映射 | 自动生成类代码框架（属性、方法声明），减少重复工作；支持逆向工程，保障设计与代码一致 | Rational Rose、StarUML |
| IDE（集成开发环境） | 集成代码编辑、编译、调试、构建的一体化环境，是CASE工具的常见实现形式 | 编辑人工补充的业务代码；调试生成的代码框架；集成版本控制，管理代码变更 | IntelliJ IDEA、Eclipse |

### 2. 工具选择的核心原则

- 匹配项目类型：不同项目（如JavaWeb、C++客户端）对应的工具组合不同（如Java项目选Eclipse+Maven，C++项目选Visual Studio）；

- 支持集成能力：工具需支持设计文件（如UML类图）导入、代码导出，以及逆向工程（代码→类图），确保设计与代码双向同步；

- 适配开发过程：工具需支撑项目选择的开发模式（如敏捷迭代需支持快速代码生成与变更）。

## 变更管理与逆向工程：保障设计与代码一致性

代码生成后难免面临需求变更或设计调整，需通过规范的变更管理和逆向工程，避免设计与代码脱节（呼应第一章“阶段评审、产品控制”原则）。

### 1. 两种变更管理方式（按需选择）

| 变更方式 | 核心逻辑 | 适用场景 | 优势 | 风险 |
| :---: | :---: | :---: | :---: | :---: |
| 仅修改代码，不更新文档 | 需求变更仅在代码中体现，设计文档保持不变 | 小型项目、敏捷迭代（快速交付优先） | 效率高，无需同步文档成本 | 长期维护困难，新成员无法通过文档理解设计 |
| 全流程同步修改 | 变更需经过“分析→设计→文档→代码”全流程，确保所有产出物一致 | 大型项目、长期维护项目（可维护性优先） | 文档与代码一致，维护成本低 | 流程繁琐，变更周期长 |

### 2. 逆向工程：一致性的核心保障

- 核心定义：将代码的修改反向映射回类图设计，或直接从代码生成设计图，实现“设计↔代码”双向同步；

- 核心作用：

  1. 当代码人工修改后（如补充业务方法），通过逆向工程更新类图，确保设计文档不落后；

  2. 无初始设计图时，可通过现有代码生成类图，快速理解系统结构（如接手legacy项目）；

- 关键要求：设计工具与编码工具需紧密集成（如StarUML生成的类图可直接导入IDEA，IDEA修改代码后可反向更新类图）。


## 单个类的代码转换：从类图到代码框架

单个类的代码生成是基础，需将第五章类图中的“类名、属性、方法、可见性”等信息，精准转化为目标编程语言的语法结构。

### 1. 类图→代码的核心转换规则

| 类图元素 | 转换规则 | 示例（Java语言，对应Employee类） |
| :---: | :---: | :---: |
| 类名 | 直接作为代码中的类名（遵循编程语言命名规范，如首字母大写） | `public class Employee { ... }` |
| 实例变量（属性） | 转化为类的成员变量，需指定类型；可见性对应访问修饰符（+→public，-→private，~→protected，*→default） | `-empno : int` → `private int empno;` |
| 方法 | 转化为类的成员方法，需指定参数类型、返回值类型；可见性对应访问修饰符 | `+getEmpno() : int` → `public int getEmpno() { return empno; }` |
| 静态属性/方法 | 类图中以下划线标注，代码中添加静态修饰符（如static） | `-empcount : int`（下划线）→ `private static int empcount;` |
| 构造方法 | 类图中显式声明的构造方法直接转换；无显式声明时，工具生成默认无参构造 | `+Employee(in firstname : String, in lastname : String)` → `public Employee(String firstname, String lastname) { ... }` |

### 2. 关键补充说明

- 方法参数类型：类图中的`in`（输入）、`out`（输出）、`inout`（输入输出）是逻辑标注，代码中仅需指定参数类型，无需保留该关键字（如`in empno : int` → `int empno`）；

- 自动生成范围：CASE工具仅生成“代码框架”（属性声明、方法声明、简单get/set方法），核心业务逻辑（如`completePctCompute()`计算进度）需人工补充；

- 静态元素慎用：静态属性/方法破坏面向对象封装性（属于“全局资源”，不依赖对象实例），仅适用于工具类、常量定义等场景（如记录文件存储路径）。

## 类关系的代码实现：关联、聚合、组合与依赖

第五章定义的类关系（关联、聚合、组合、依赖），需通过特定的代码结构实现，核心是“通过成员变量或方法参数体现关系”，兼顾封装性与逻辑正确性。

### 1. 关联关系的代码实现（核心：成员变量+多重性）

关联关系是类间的静态“拥有”关系，代码中通过成员变量体现，多重性决定成员变量的类型（单个对象/集合）。

#### （1）不同多重性的关联实现

| 多重性 | 核心逻辑 | 代码示例（ProjectTask与Employee的关联） |
| :---: | :---: | :---: |
| 0..1（可选唯一） | 成员变量可为null，构造时无需初始化，通过set方法后续赋值 | `private Employee director;`（默认null，不允许显式赋值为null） |
| 1（强制唯一） | 成员变量必须有值，构造时初始化或赋值 | `private Employee director = new Employee();` 或 构造方法中赋值 |
| *（任意多个） | 成员变量为集合类型（需结合元素顺序、唯一性选择集合） | `private List<Employee> director;`（Java中用List实现有序可重复集合） |

#### （2）集合类型选择指南（匹配多重性与业务需求）

| 元素顺序要求 | 元素唯一性要求 | 集合类型（Java） | 适用场景 |
| :---: | :---: | :---: | :---: |
| 不要求 | 无重复（{unique}） | Set（如HashSet） | 项目任务的负责人不允许重复 |
| 不要求 | 允许重复 | Bag（或ArrayList） | 项目参与人员可重复（如兼职多次参与） |
| 要求（{ordered}） | 无重复（{unique}） | OrderedSet（如TreeSet） | 按入职顺序排列的项目成员，不重复 |
| 要求（{ordered}） | 允许重复 | List（如ArrayList） | 按任务分配顺序排列的参与人员，可重复 |

### 2. 聚合与组合：关联关系的特殊形式（强/弱归属）

聚合与组合都是“整体-部分”的关联关系，核心区别在于“部分是否依赖整体存在”，代码实现仅在生命周期管理上有差异。

| 关系类型 | 核心逻辑（生命周期） | 代码示例（ProjectTask与Employee） | 业务场景解释 |
| :---: | :---: | :---: | :---: |
| 聚合（Aggregation） | 部分可独立于整体存在（员工可参与多个项目任务） | `private Employee director;`（Employee对象可被多个ProjectTask引用） | 员工是独立实体，可同时负责多个项目任务，任务删除不影响员工存在 |
| 组合（Composition） | 部分依赖整体存在（员工是任务的专属合同，任务删除则员工合同失效） | `private Employee director = new Employee();`（在ProjectTask构造方法中创建Employee，外部无法引用） | 员工此处指“任务专属合同”，任务结束（对象删除），合同自动失效（Employee对象被回收） |

### 3. 依赖关系：瞬时交互（非静态关联）

依赖关系是类间的“瞬时访问”关系（如方法参数传递），代码中通过“方法参数、局部变量”体现，而非成员变量（区别于关联关系）。

| 关系特征 | 代码实现方式 | 示例（Bicycle与Pump的依赖） |
| :---: | :---: | :---: |
| 瞬时交互 | 依赖类作为被依赖类方法的参数，仅在方法执行时存在关联 | Bicycle类的expand方法接收Pump参数：<br>`public class Bicycle {`<br>`  public void expand(Pump pump) {`<br>`    pump.blow(); // 仅在方法执行时调用Pump的功能`<br>`  }`<br>`}` |
| 无静态关联 | 被依赖类不作为成员变量，方法执行结束后关联消失 | 无Bicycle类的成员变量指向Pump，仅在充气时临时传入Pump对象 |

### 4. 关联关系的封装性优化（避免破坏封装）

直接通过成员变量暴露关联对象（如`private Employee director;`），可能导致外部类直接调用关联对象的方法（如`task.getDirector().setLastname("xxx")`），破坏封装性。优化方案：

- 方案1：使用接口屏蔽多余方法（“棒棒糖”设计）：定义仅包含必要方法的接口（如TaskDirector），关联对象声明为接口类型，而非具体类（如`private TaskDirector director;`）；

- 方案2：通过宿主类提供间接修改方法：在ProjectTask中提供`updateLastNameDirector(String lastname)`，内部调用Employee的setLastname方法，外部无法直接操作Employee对象（聚合关系常用）。

## 架构模式落地：MVC的代码实现

第四章提到的MVC（模型-视图-控制器）架构模式，需通过类的代码组织实现，核心是“分离数据（Model）、展示（View）、交互逻辑（Controller）”，呼应第一章“高内聚、低耦合”原则。

### 1. MVC代码实现的核心结构（Java示例）

#### （1）类图设计（关联第五章类设计）

| 组件类型 | 核心类/接口 | 核心职责 | 代码示例片段 |
| :---: | :---: | :---: | :---: |
| Model（模型） | XModel类 + XModelListener接口 | 存储业务数据，提供数据修改方法，通知视图更新 | `public class XModel {`<br>`  private int modelValue = 50;`<br>`  private List<XModelListener> listeners = new ArrayList<>();`<br>`  // 注册视图监听器`<br>`  public void addXModelListener(XModelListener listener) {`<br>`    listeners.add(listener);`<br>`  }`<br>`  // 修改数据并通知视图`<br>`  public void changeValue(int delta) {`<br>`    modelValue += delta;`<br>`    fireXModelChanged(); // 通知所有注册的视图`<br>`  }`<br>`}` |
| View（视图） | XView类（实现XModelListener） | 展示数据，响应模型更新通知 | `public class XView implements XModelListener {`<br>`  private XModel xmodel;`<br>`  @Override`<br>`  public void xModelChanged() {`<br>`    // 模型数据变化，更新视图展示（如滑块位置）`<br>`    System.out.println("当前值：" + xmodel.getValue());`<br>`  }`<br>`}` |
| Controller（控制器） | XController类 | 接收用户输入，调用模型修改数据 | `public class XController {`<br>`  private XModel xmodel;`<br>`  public XController(XModel xmodel) {`<br>`    this.xmodel = xmodel;`<br>`  }`<br>`  // 响应“加”按钮点击`<br>`  public void plus() {`<br>`    xmodel.changeValue(1);`<br>`  }`<br>`  // 响应“减”按钮点击`<br>`  public void minus() {`<br>`    xmodel.changeValue(-1);`<br>`  }`<br>`}` |

### 2. MVC的灵活调整（不局限于固定模板）

MVC的核心是“分离职责”，实际开发中可根据场景调整：

- 无多视图时：无需定义XModelListener接口，视图直接引用模型，定期查询数据更新；

- 单视图场景：模型的listener可改为单个视图引用（而非集合），简化代码；

- 控制器与视图合并：简单场景中（如按钮同时负责交互和显示结果），可让控制器兼具视图功能（如JButton的setText方法显示结果）。

## 多视图验证：非功能需求的代码支撑

第五章类图主要支撑功能性需求，非功能性需求（性能、可用性、部署）需通过UML“4+1视图”的代码落地来保障，确保系统全面满足第三章需求。

### 1. 4+1视图的代码落地方式

| 视图类型 | 核心关注（非功能需求） | 代码/配置实现方式 | 示例 |
| :---: | :---: | :---: | :---: |
| 进程视图 | 性能、并发、可用性（如多线程处理） | 实现多线程（如Java继承Thread或实现Runnable），用UML构造型<<thread>>标注活动类 | `public class ProjectThread implements Runnable {`<br>`  @Override`<br>`  public void run() { /* 并发处理项目数据 */ }`<br>`}` |
| 开发视图 | 代码组织、复用、可维护性（如构件拆分） | 将类按功能打包（如第四章包结构），生成构件（JAR包），明确接口依赖 | 项目类打包为pm.jar，员工类打包为employee.jar，通过接口AllProjects实现依赖 |
| 物理视图 | 部署、硬件映射、网络通信（如分布式部署） | 编写部署配置文件，指定构件在硬件节点的部署位置，配置网络连接 | 部署图中Database Server部署projectdata.jar，Project Management Server部署pm.jar，通过LAN/1Gbit网络通信 |
| 用例视图（+1） | 功能覆盖、场景可行性 | 通过顺序图/通信图验证代码调用流程，确保用例场景可实现 | 项目创建场景的顺序图对应代码中ProjectMan的addProject方法调用 |

### 2. 核心价值

多视图的代码落地，确保系统不仅“能实现功能”，还能满足第三章的非功能性需求（如性能、可部署性），是工程化开发的必要环节——避免“功能达标但部署失败”“代码正确但并发性能不足”等问题。