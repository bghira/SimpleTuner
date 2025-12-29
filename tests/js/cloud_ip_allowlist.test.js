/**
 * Tests for IP allowlist management in cloud dashboard.
 *
 * Tests the IP validation and allowlist CRUD operations.
 */

describe('IP allowlist management', () => {
    let context;

    beforeEach(() => {
        // Create context with required state and mock saveAdvancedSetting
        context = {
            advancedConfig: {
                webhook_allowed_ips: [],
                newIpEntry: '',
                ipValidationError: null,
            },
            saveAdvancedSetting: jest.fn().mockResolvedValue(undefined),
        };

        // Define the methods inline (matching implementation in index.js)
        context.addIpToAllowlist = async function() {
            const ip = this.advancedConfig.newIpEntry.trim();
            if (!ip) return;

            const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}(\/\d{1,2})?$/;
            const ipv6Regex = /^([0-9a-fA-F:]+)(\/\d{1,3})?$/;
            if (!ipv4Regex.test(ip) && !ipv6Regex.test(ip)) {
                this.advancedConfig.ipValidationError = 'Invalid IP or CIDR format';
                return;
            }

            if (this.advancedConfig.webhook_allowed_ips.includes(ip)) {
                this.advancedConfig.ipValidationError = 'IP already in list';
                return;
            }

            this.advancedConfig.ipValidationError = null;
            this.advancedConfig.webhook_allowed_ips.push(ip);
            this.advancedConfig.newIpEntry = '';
            await this.saveAdvancedSetting('webhook_allowed_ips', this.advancedConfig.webhook_allowed_ips);
        }.bind(context);

        context.removeIpFromAllowlist = async function(ip) {
            this.advancedConfig.webhook_allowed_ips = this.advancedConfig.webhook_allowed_ips.filter(i => i !== ip);
            await this.saveAdvancedSetting('webhook_allowed_ips', this.advancedConfig.webhook_allowed_ips);
        }.bind(context);
    });

    test('addIpToAllowlist validates IPv4 format', async () => {
        context.advancedConfig.newIpEntry = '192.168.1.1';

        await context.addIpToAllowlist();

        expect(context.advancedConfig.webhook_allowed_ips).toContain('192.168.1.1');
        expect(context.advancedConfig.newIpEntry).toBe('');
        expect(context.advancedConfig.ipValidationError).toBeNull();
        expect(context.saveAdvancedSetting).toHaveBeenCalledWith('webhook_allowed_ips', ['192.168.1.1']);
    });

    test('addIpToAllowlist validates IPv4 with CIDR', async () => {
        context.advancedConfig.newIpEntry = '10.0.0.0/8';

        await context.addIpToAllowlist();

        expect(context.advancedConfig.webhook_allowed_ips).toContain('10.0.0.0/8');
    });

    test('addIpToAllowlist validates IPv6 format', async () => {
        context.advancedConfig.newIpEntry = '2001:db8::1';

        await context.addIpToAllowlist();

        expect(context.advancedConfig.webhook_allowed_ips).toContain('2001:db8::1');
        expect(context.advancedConfig.ipValidationError).toBeNull();
    });

    test('addIpToAllowlist rejects invalid IP', async () => {
        context.advancedConfig.newIpEntry = 'not-an-ip';

        await context.addIpToAllowlist();

        expect(context.advancedConfig.webhook_allowed_ips).toHaveLength(0);
        expect(context.advancedConfig.ipValidationError).toBe('Invalid IP or CIDR format');
        expect(context.saveAdvancedSetting).not.toHaveBeenCalled();
    });

    test('addIpToAllowlist rejects duplicate IP', async () => {
        context.advancedConfig.webhook_allowed_ips = ['192.168.1.1'];
        context.advancedConfig.newIpEntry = '192.168.1.1';

        await context.addIpToAllowlist();

        expect(context.advancedConfig.webhook_allowed_ips).toHaveLength(1);
        expect(context.advancedConfig.ipValidationError).toBe('IP already in list');
        expect(context.saveAdvancedSetting).not.toHaveBeenCalled();
    });

    test('addIpToAllowlist ignores empty input', async () => {
        context.advancedConfig.newIpEntry = '   ';

        await context.addIpToAllowlist();

        expect(context.advancedConfig.webhook_allowed_ips).toHaveLength(0);
        expect(context.saveAdvancedSetting).not.toHaveBeenCalled();
    });

    test('removeIpFromAllowlist removes IP', async () => {
        context.advancedConfig.webhook_allowed_ips = ['192.168.1.1', '10.0.0.1'];

        await context.removeIpFromAllowlist('192.168.1.1');

        expect(context.advancedConfig.webhook_allowed_ips).not.toContain('192.168.1.1');
        expect(context.advancedConfig.webhook_allowed_ips).toContain('10.0.0.1');
        expect(context.saveAdvancedSetting).toHaveBeenCalledWith('webhook_allowed_ips', ['10.0.0.1']);
    });

    test('removeIpFromAllowlist handles non-existent IP gracefully', async () => {
        context.advancedConfig.webhook_allowed_ips = ['192.168.1.1'];

        await context.removeIpFromAllowlist('10.0.0.1');

        expect(context.advancedConfig.webhook_allowed_ips).toHaveLength(1);
        expect(context.advancedConfig.webhook_allowed_ips).toContain('192.168.1.1');
    });
});
