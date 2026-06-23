# Experiment Console

学習ログの閲覧と `src/CONSTANTS.py`、各モデルの `constants.py` の変更をブラウザから行うWebアプリ。

Constants画面では、現在のGitブランチとローカル変更数を確認し、`git pull --ff-only` でGitHubの変更を取得できる。

## ローカル起動

プロジェクトルートで実行する。

```bash
./mlp-web
```

ブラウザで <http://127.0.0.1:8765> を開く。

ポートを変更する場合は、CLIオプションを渡す。

```bash
./mlp-web --port 9000
```

利用可能なオプションは次で確認できる。

```bash
./mlp-web --help
```

## GPUサーバーで起動

サーバー側では外部公開せず、標準のループバックアドレスで起動する。

```bash
cd ~/making-mlp
./mlp-web
```

手元のMacからSSHポートフォワードする。

```bash
ssh -L 8765:127.0.0.1:8765 student222@kamiyama-server
```

接続中にMacのブラウザで <http://127.0.0.1:8765> を開く。

## 注意

- Webサーバーに認証機能はないため、GPUサーバーで `--host 0.0.0.0` を使わない。
- constantsの変更は次回開始する学習から反映される。
- 学習プロセスの起動中はconstantsを変更しない。
- ファイルが外部で変更された場合、古い画面からの保存は拒否される。再読み込みしてから編集する。
- Pullはローカル変更を自動でstash・破棄しない。競合する変更がある場合は失敗内容を画面に表示する。
- Webアプリ自身が更新された場合は、Pull後に `./mlp-web` を再起動する。
