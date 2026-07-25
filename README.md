# beian

> Query ICP beian like whois

[![vibed](https://img.shields.io/badge/100%25-vibed-blue)](https://github.com)

Query ICP registration info from MIIT's official system. Solves CAPTCHA automatically.

## Install

```bash
uvx beian --help
```

Or install permanently:

```bash
uv tool install beian
```

## Usage

```bash
# ICP filing (default)
beian baidu.com
beian "北京百度网讯科技有限公司"

# ICP license (增值电信业务经营许可证)
beian --license "小米科技有限责任公司"

# App / miniprogram / quick app
beian --app "微信"
beian --miniprogram "人民法院在线服务"
beian --quickapp "计算器"

# Batch query
beian baidu.com qq.com taobao.com

# JSON output
beian --raw baidu.com

# Screenshot (ICP filing only)
beian --screenshot baidu.com

# Verbose progress
beian -v baidu.com

# Custom retry count
beian --retry 10 "小米科技有限责任公司"
```

## Query Types

| Flag | Type | Source |
|------|------|--------|
| (default) | ICP filing — website (备案) | beian.miit.gov.cn |
| `--app` | ICP filing — app (备案) | beian.miit.gov.cn |
| `--miniprogram` | ICP filing — miniprogram (备案) | beian.miit.gov.cn |
| `--quickapp` | ICP filing — quick app (备案) | beian.miit.gov.cn |
| `--license` | ICP license (许可证) | tsm.miit.gov.cn |

## ICP License Features

The `--license` query automatically fetches:
- License info (许可证信息)
- Business types and scope (业务种类及覆盖范围)
- Authorization info (授权信息)
- Annual report (年报公示)

## How It Works

**ICP filing**: Uses Playwright to control a headless browser, solves slider CAPTCHA via gap detection algorithm (~67% first-try success rate).

**ICP license**: Uses `ddddocr` with edge enhancement preprocessing to solve text CAPTCHA (~12% per attempt, ~100% with retries).

## Development

```bash
uv sync
uv run python -m beian.cli --help
```

## License

GPL 2.0 or later
