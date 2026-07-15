# 📊 Data Dictionary - Engagement & Retention Dataset

> **Accenture Project Cycle Datathon** · a synthetic dataset modelling a SaaS product's customers and
> their behaviour. The prediction target is **customer churn** (`churned_30d` / `churned_90d`).

Four tables, joinable on **`user_id`**:

| Table | One row per… | Purpose |
|-------|--------------|---------|
| [`users.csv`](#-userscsv) | customer | Customer profile **+ the churn labels to predict** |
| [`sessions.csv`](#-sessionscsv) | login session | How and when customers log in |
| [`events.csv`](#-eventscsv) | in-app action | What customers do inside the product |
| [`billing.csv`](#-billingcsv) | customer-month | Subscription, revenue, and support signals |

---

## 👤 users.csv
*One row per customer - this is where the prediction targets live.*

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier **(join key)** |
| `signup_date` | Date when the user signed up |
| `plan_tier` | Subscription plan (free / standard / premium) |
| `company_size` | Size category of the user's company (1-10 … 200+) |
| `region` | Geographic region |
| `industry` | Business industry category |
| `acquisition_channel` | How the user was acquired |
| `is_enterprise` | Whether the user is an enterprise customer |
| 🎯 `churned_30d` | **Target** - user churned within 30 days |
| 🎯 `churned_90d` | **Target** - user churned within 90 days |
| `downgraded` | User downgraded their plan |
| `expansion_event` | User expanded their usage |

---

## 🖥️ sessions.csv
*One row per login session.*

| Column | Description |
|--------|-------------|
| `session_id` | Unique session identifier |
| `user_id` | User identifier **(join key)** |
| `session_start` / `session_end` | Session start and end timestamps |
| `device` | Device type used |
| `os` | Operating system |
| `app_version` | Application version |
| `country` | Country code |
| `session_length_sec` | Session duration in seconds |

---

## ⚡ events.csv
*One row per in-app action.*

| Column | Description |
|--------|-------------|
| `event_id` | Unique event identifier |
| `user_id` | User identifier **(join key)** |
| `session_id` | Session identifier |
| `ts` | Event timestamp |
| `feature_name` | Feature being used |
| `action` | Type of action performed |
| `duration_ms` | Event duration in milliseconds |
| `latency_ms` | Response latency in milliseconds |
| `success` | Whether the event completed successfully |

---

## 💳 billing.csv
*One row per customer per billing month.*

| Column | Description |
|--------|-------------|
| `user_id` | User identifier **(join key)** |
| `month` | Billing month |
| `plan_tier` | Current subscription plan |
| `active_seats` | Number of active seats |
| `mrr` | Monthly recurring revenue |
| `discount_applied` | Whether a discount was applied |
| `invoices_overdue` | Whether invoices are overdue |
| `support_ticket_count` | Number of support tickets |

---

> [!TIP]
> The starter notebooks only use **`users.csv`**. Aggregating `sessions`, `events`, and `billing` per
> user (join on `user_id`) is where the strongest churn signals usually hide - see the
> [Info Pack](../../INFO_PACK.md#-datathon-tips) for ideas.
