# Spec: delivery-pipeline

## Objective

Tự động hóa toàn bộ đường đi từ commit đến image public trên Docker Hub.

- CI: mỗi push hoặc PR được lint và test.
- CD: khi `main` xanh, build hai image, smoke test, rồi push lên `quanluuu/deepsat-api` và `quanluuu/deepsat-demo`.

Acceptance criteria:

1. Push lên `main` hoặc mở PR vào `main` sẽ chạy workflow `ci.yml` gồm các job `lint`, `test-demo`, `test-api`. Bất kỳ job nào fail thì workflow đỏ.
2. Job `docker` chỉ chạy sau khi ba job trên xanh. Job này tải weights từ GitHub Release tag `weights-v1` bằng `gh release download`. Sau khi tải, job verify `weights/SHA256SUMS`; sai checksum thì job đỏ.
3. Trên PR: job `docker` build cả hai image và chạy smoke test (`/health`, `scripts/smoke_api.py`), nhưng không push.
4. Trên push `main`: sau smoke test, job push hai image với tag `latest` và `sha-<7 ký tự commit>`.
5. Trên push tag `v*.*.*` (ví dụ `v1.0.0`): job push thêm tag `1.0.0`.
6. Login Docker Hub dùng secret `DOCKERHUB_USERNAME` và `DOCKERHUB_TOKEN` (Access Token, quyền Read & Write). Không dùng mật khẩu tài khoản.
7. Khi cache còn, lint + test chạy dưới 5 phút, job docker chạy dưới 15 phút.
8. Từ máy bất kỳ có Docker, `docker pull quanluuu/deepsat-api:latest` chạy được mà không cần login.

## Tech Stack

GitHub Actions `ubuntu-latest`. Các action dùng:

- `actions/checkout@v4`
- `actions/setup-python@v5`
- `docker/setup-buildx-action@v3`
- `docker/login-action@v3`
- `docker/metadata-action@v5`
- `docker/build-push-action@v6` với `cache-from/cache-to: type=gha`

## Commands

```bash
# 1 lần: tạo Release chứa weights (máy local, cần gh CLI)
sha256sum best.pth dinov3_vits16_pretrain_lvd1689m-08c60483.pth > weights/SHA256SUMS
gh release create weights-v1 best.pth dinov3_vits16_pretrain_lvd1689m-08c60483.pth --title "Model weights v1" --notes "TSViT_DINOv3 fold1 T9"

# 1 lần: secrets cho Actions
gh secret set DOCKERHUB_USERNAME --body quanluuu
gh secret set DOCKERHUB_TOKEN      # dán Access Token khi được hỏi

# phát hành bản có version
git tag v1.0.0 && git push origin v1.0.0

# theo dõi
gh run list --workflow ci.yml
gh run watch
```

## Project Structure

```text
+ .github/workflows/ci.yml   -> lint, test-demo, test-api, docker (build + smoke + push có điều kiện)
+ weights/SHA256SUMS         -> commit vào git; bản thân weights không commit
  .gitignore                 -> thêm weights/*.pth, weights/dinov3/, *.part
```

Snapshot `facebookresearch/dinov3` được clone trong job `docker` theo commit SHA đã pin, vào `weights/dinov3`.

## Code Style

Workflow chia job rõ ràng. Mỗi step có `name`. Mỗi job khai báo `permissions` tối thiểu. Điều kiện push viết một chỗ:

```yaml
push: ${{ github.event_name == 'push' }}
```

## Testing Strategy

| Kiểm tra | Cách |
|---|---|
| Pipeline bắt lỗi lint | PR thử có lỗi ruff, job `lint` đỏ |
| Pipeline bắt lỗi logic | PR thử sửa `status_of` sai, job `test-api` đỏ |
| CD chạy đúng | Merge lên `main`, xem tag `latest` và `sha-...` trên Docker Hub |
| Image chạy được | `docker pull` + `docker run` trên máy chưa có code |

## Boundaries

- Always: secret chỉ đọc qua `${{ secrets.* }}`; pin action theo major version; verify checksum weights.
- Ask first: push public lần đầu (xem Open Question license trong SPEC.md); thêm trigger hoặc job mới; bật branch protection.
- Never: in token ra log; commit weights; dùng `continue-on-error` để vượt qua bước lỗi; push image khi smoke test fail.

## Success Criteria

- Ba lần chạy Actions được ghi lại: một PR đỏ, một PR xanh không push, một push `main` có push image.
- Hai repo Docker Hub public có tag `latest`.

## Open Questions

1. Có tự cập nhật phần mô tả repo trên Docker Hub (README) bằng `peter-evans/dockerhub-description` không? Đề xuất: không, để ít phụ thuộc.
