import torch
print(f"PyTorch 版本: {torch.__version__}")

# 测试 pyg-lib 是否可用
try:
    import pyg_lib
    print(f"pyg-lib 已安装: {pyg_lib.__version__}")
    print("pyg-lib 可用 ✓")

    # 测试是否能实际调用采样函数
    try:
        from torch_geometric.loader import NeighborLoader

        # 尝试创建一个简单的测试
        loader = NeighborLoader(
            cora_data,
            num_neighbors=[5, 3],
            batch_size=32,
            input_nodes=cora_data.train_mask
        )

        # 尝试获取一个批次
        batch = next(iter(loader))
        print(f"\nNeighborLoader 测试成功 ✓")
        print(f"  原始图: {cora_data.num_nodes} 节点")
        print(f"  采样子图: {batch.num_nodes} 节点")
        print(f"  批次大小: {batch.batch_size}")

    except Exception as e:
        print(f"\nNeighborLoader 测试失败: {e}")
        print("pyg-lib 已安装但版本可能不匹配")

except ImportError as e:
    print(f"pyg-lib 未安装: {e}")
    print("\n建议:")
    print("  对于 Cora 这种小图，不需要采样")
    print("  或安装匹配版本: pip install pyg-lib -f https://data.pyg.org/whl/torch-2.1.0+cpu.html")
